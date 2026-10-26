# 一次 6B VLA 全参微调的踩坑全记录

> 在 AMD Radeon Cloud（8×W7900D 48GB / ROCm 7.2.1）上，把 LingBot-VLA 2.0 6B 在 RoboTwin 2.0 的 50 个 Aloha-AgileX clean 任务上做全参 SFT，从零到提交。
>
> 这份文档只记录**真实发生过的问题**、**当时看到的症状**、**定位过程**和**最终解法**。所有凭据、密钥、token 均已剔除。

---

## 0. 结果先说

| 项 | 值 |
|---|---|
| 训练 | 1500 步 ≈ 2.8 epoch，损失 0.0555 → 0.020 |
| 实测成功率 | clean 32 任务 / 192 episodes = **43.2%** |
| 交付物 | HF 全量权重（19 文件 / 25,519,988,667 字节）+ `results.json` + 代码 + 复现报告 |
| 最终状态 | 已提交，待评测 |

**诚实结论**：能跑通、能复现、能提交；但训练量不足（2.8 epoch），长程/双手协同任务成功率为 0，泛化（randomized）未充分验证。**不是争冠水平。**

---

## 一、AMD ROCm 环境类

### 1.1 `/tmp` 是 4GB tmpfs，AMD COMGR 会把它写爆

**症状**：8 个 rank 一起挂，日志里只有 `torch.distributed.elastic` 的 `ChildFailedError` 外壳，看不到 Python 堆栈。

**真因**：日志中段有
```
LLVM ERROR: IO failure on output stream: No space left on device
```
AMD 的 COMGR 编译器每个进程在 `/tmp` 写约 520MB，8 rank 并发 = 4.2GB > 4GB tmpfs。

**修复**：
```bash
export TMPDIR=/root/tmp            # 指到大盘
export TRITON_CACHE_DIR=/root/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/root/torchinductor
```

**教训**：`ChildFailedError` 全 rank 齐挂 + 无 Python 堆栈时，**先看有没有 LLVM/编译器层错误**，不要只盯 Python traceback。

### 1.2 `aiter` 导入失败导致 flash-attention 不可用

镜像没注入环境变量，`aiter` 要 triton≥3.6.0 而实际 3.5.1：
```
export AITER_TRITON_ONLY=1                      # 这是硬依赖，不是优化
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 1.3 NFS 上不能用 `rsync -a`，也不能用 `--no-owner` 兜底

**症状**：
```
rsync: [receiver] chown "/workspace/.../.safetensors.XXX" failed: Operation not permitted (1)
rsync error: some files/attrs were not transferred (code 23)
```

**真因**：持久盘是 NFS，做了 root-squash（属主显示 `nobody:nogroup`）。root 跑 `rsync -a` 会尝试 chown → EPERM。

**关键坑**：`--no-owner --no-group --no-perms` **压不住**。小文件手工测试能过，大文件必挂 —— 极难归因。

**可用方案**：`cp -r`（`cp -a` 对属主只报 warning 并继续，退出码 0）。

### 1.4 平台会给"降级实例"

请求 8 卡 + Devzone 模型挂载，界面摘要写着 `8 GPU · 128 核 · 440GB`、启动闸门全绿，**实际只透传了 1 张卡**（`/dev/dri` 只有 `renderD128`），`/models` 是空目录。按 1 卡档计费，CPU/内存却是 8 卡档。

**教训**：启动后**必须 SSH 进去验证实际资源**，不能只看界面。

### 1.5 容器没有正经 init，僵尸进程无人回收

```
PID 1 = jupyter-lab（不 wait 子进程）
→ 每次训练失败/重启都留一批僵尸，累积到 890 个
```
僵尸只占 PID 槽位（`pid_max=4194304`），不占 CPU/内存/显存，但**负载均值会被误导**（EWMA 残值）。

### 1.6 `pkill -f <pattern>` 会杀掉自己

pattern 若出现在自己的命令行里（如 `train_lingbotvla`），`pkill -f` 自我匹配 → 整条远端命令静默死掉，没有任何输出。

**修复**：把启动/清理逻辑写进远端脚本文件再执行；或用 `[t]asks.vla.train` 这种字符类转义绕过。

---

## 二、训练与 checkpoint 类

### 2.1 上游键名不是想当然的那个

```bash
--set train.num_train_steps=1500
# → ValueError: Some specified arguments are not used by the ArgumentParser:
#   ['--train.num_train_steps', '1500']
```
正确键是 `train.max_steps` / `train.save_steps` / `train.lr` / `train.lr_min`。

### 2.2 官方 `lr_min` 比我们要设的 `lr` 还大，余弦退火会反向

官方模板：`lr: 1.0e-4`、`lr_min: 5.0e-5`。
若把 `lr` 改成 `2e-5` 而不管 `lr_min`，退火会**从 2e-5 涨到 5e-5**。**改 lr 必须同时改 lr_min。**

### 2.3 改训练超参必须重启，而重启会丢掉整个 save 区间 ★

`save_steps=100`，checkpoint 每 100 步才落盘。
在 **step 1139** 重启想改 `max_steps`，最新 checkpoint 是 `global_step_1100` → **白丢 39 步 ≈ 1 小时 ≈ 8 credits**。

**硬规则**：**只允许在 checkpoint 边界（`global_step` 为 `save_steps` 整数倍）重启。**

### 2.4 DataLoader worker 数必须与 checkpoint 里保存的一致

```
AssertionError: self._num_workers == state_dict[self._NUM_WORKERS]
```
torchdata 的 StatefulDataLoader 要求恢复时的 worker 数与保存时相同。
备份里存的是 `_num_workers=2`，新配置写 8 → 直接断言失败。

**另一个延申坑**：把 `extra_state` 目录挪走后，恢复变成 `FileNotFound` →
**日志里出现 `Starting training from scratch`，不报错、只是默默从头训**。
这是个**静默降级**，必须显式 grep 这一行。

### 2.5 HF 权重是异步写的，同步器会抢跑

日志：`[async_hf] saving HF checkpoint for global_step_400`
同步循环在 12:28:42 扫到它时，HF 还在写 → 源只有 50,206 MB（写完应为 74,543 MB）。

**危险**：如果校验只查"有没有分片"而不查"齐不齐"，**残缺源与残缺副本能比对一致而误判通过**。

**修复**：精确数量校验 —— `model` 128 distcp · `optimizer` 128 · `hf_ckpt` 17 · `extra_state` 8，任一不符即拒绝。

### 2.6 训练 OOM 的真凶可能不是训练自己

报错：
```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 50.00 MiB.
GPU 1 ... 0 bytes is free. Of the allocated memory 23.73 GiB is allocated by PyTorch
```
**"0 bytes free" 和 "只分配了 23.73 GiB" 是矛盾的** —— 说明显存被别的进程占着。

真因：评测的模型服务 `deploy.lingbot_vla_v2_policy` 是 `setsid` 分离启动的，
**杀父进程 `mayi.algorithm evaluate` 杀不到它们** → 8 个孤儿各持 25.9 GB。

**修复**：显式杀 `deploy.lingbot_vla_v2_policy` / `run_*_benchmark`，并**等 `rocm-smi` 显存回落到 <2GB** 才起训练。判据看绝对字节数，不看进程数。

---

## 三、持久盘与备份类（代价最大）

### 3.1 事故：跑到 1200 步，持久盘只有 200 步

**损失**：1000 步 = 26.4 GPU 小时 ≈ 211 credits ≈ 总额度 29%。

**原因**：整个运行期**没有任何自动同步**。只在开始时手工拷了一次，
之后 1000 步全靠"我记得再拷一次"——而没记得。

### 3.2 第二个成因：没校验就删源

腾空间时执行了 `rm -rf` 删掉持久盘上唯一的 DCP，依据只是"根盘应该有一份拷贝"——**没校验就删源**。

**硬规则**：腾空间的顺序是绝对的 —— **先证明新副本可用（校验），再删旧副本。**

### 3.3 100GB 持久盘的物理约束

单个 DCP = 72.8GB，持久盘 100GB ⇒ **只能放一份**。
所以覆盖同步必然出现"**先删旧副本才能写新副本**"的窗口（实测约 6 分钟）。那 6 分钟若实例挂掉，只能从根盘恢复。

**这是绕不过去的物理约束**，除非改用外部对象存储。

### 3.4 平台销毁实例时，持久盘会活下来（重要发现）

积分耗尽被平台**自动销毁**后重建实例，`/workspace` 上的
`dcp-roll`（73G，回执 step=1500）+ `hf-run…step1500`（24G）**完好无损**。

**结论**：持久盘 PVC 跨实例存活，是真正的灾备层。但这**不能替代"拷回本地"** —— 实例一旦彻底销毁，PVC 也可能一起消失。

### 3.5 同步脚本的三个毁数据缺陷

早期版本：
1. `rsync -a --delete` 直写唯一目标目录 → 中断即损坏唯一副本
2. `ls -td` 按 **mtime** 选最新 → 不是步号最大
3. `.synced` 只记 step 无 run_id → 重训同 step 会被判"已同步"而静默跳过
4. **它还会主动抵消清理**：删掉目标目录后 60 秒内又被重新造出来占满磁盘

**修复后的正确做法**：
- 暂存代 → 逐文件 SHA256 + 文件数 + 字节三项校验 → `mv` 原子改名 → `ln -sfn` 切指针
- 版本名含 `run_id`：`hf-<run_id>-step<N>`
- 回执记录 `run_id / step / files / bytes / manifest_sha256 / 时间`
- **按步号数值排序**发现 checkpoint，不按 mtime

---

## 四、评测类

### 4.1 官方启动器硬性要求 4 或 8 卡

```
mayi.algorithm evaluate: --num-gpus 1:
入口的 --gpu-count 只接受 4/8（它自己的 choices）
```
`run_clean_benchmark.py` 里是 `choices=(4, 8)` 硬编码。
降级实例只有 1 卡时，**必须先改这行**才能评测。改了要在复现报告里披露。

### 4.2 评测缺 `lingbotvla_cli.yaml` 会 fail-closed

```
no lingbotvla_cli.yaml at <output>; the deploy policy cannot load model config unconditionally
```
这个文件是**训练启动时**写进 `<output>/` 的，随实例销毁一起没了。

**不能随便从镜像复制那份**：镜像版是 `use_lora: true` + 镜像自己的混合 `norm_stats`，
与全参 SFT + clean-only 重算统计**不符** —— 拿它评测，数字不可信。

### 4.3 1 卡评测还需要正确的 norm_stats

镜像版的 `norm_stats_file` 指向 `assets/norm_stats/robotwin.json`（含 randomized 的混合统计）。
正确版指向 `.../stats/clean50.json`（clean-only 重算，sha256 可校验）。
两者只差 4 行，但**用错就违反"必须重算"的合规要求**。

### 4.4 评测时显存贴满整卡

实测 **51.37 GB / 51.5 GB**（48GB 卡的十进制值）。
所以**评测与训练不能同卡并行**，必须串行。

### 4.5 「loss 平台期」不等于「模型变好」

```
step 200 → 900    0.0555 → 0.0254   降 54%   快速收敛期
step 900 → 1500   0.0254 → 0.020    降 20%   平台期
```
但实测成功率从 step≈1000 的 42.9% 到 step1500 的（1卡）58.8% —— **loss 平了，成功率还能涨**。
**不能用 loss 判断模型质量**，必须实测。

---

## 五、上传与提交类（最折磨的一段）

> 目标：把一个 25.5GB 的 zip 传到天池的 OSS，STS token 只有 1 小时有效期。

### 5.1 Windows 自带 `tar` 的 zip 写入有 4GB 墙

```
tar -a -c -f out.zip -C dir .      # 660MB 的输入 → 秒出
tar -a -c -f out.zip -C dir .      # 4.9GB 的输入 → 卡在 0 字节，5 分钟不动
```
bsdtar 的 zip 写入**不支持 ZIP64**，超过 4GB 就静默卡死。

**修复**：用 Python 的 `zipfile`（`allowZip64=True`）——
25.5GB 打成 zip **只要 59 秒**：
```python
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
    for dp, _, fn in os.walk(root):
        for f in fn:
            full = os.path.join(dp, f)
            z.write(full, os.path.relpath(full, root))
```
（`ZIP_STORED` 不压缩 —— safetensors 本就压不动，省时间）

### 5.2 `cmd /c` 起的进程会随 SSH 会话断开而死 ★

最坑的一个。表现：
- 我 `nohup ssh win 'cmd /c E:\upload.cmd'` 启动上传
- 看起来在传（拿到 0.25% → 0.5% → 1.25%）
- **我的 SSH 被 `timeout` 杀掉后，Windows 上的 ossutil 也跟着死了**
- 日志空、进程消失，但**我以为是网络问题，重启了三四次**

**修复**：`start /b cmd /c xxx`（脱离会话）或 `schtasks`（计划任务）。

### 5.3 SSH 命令通道会被会话数打满

Windows OpenSSH 有 `MaxSessions`（默认 10）。
我之前用 `timeout` 杀了几十个 ssh 进程，它们留下半开连接没释放 →
**新连接能建立，但命令通道发不出去，返回空**。

**极其误导的现象**：
```
ssh win 'echo HELLO'          → HELLO          （偶尔能通）
ssh win 'type E:\big.log'     → （空）          （大输出必挂）
scp win:E:/big.log .          → 成功            （scp 走不同通道，不受影响）
```
我因此**多次误判"进程不在运行"**，而实际上传一直在跑。

**修复**：
```bash
ssh -o ControlMaster=auto -o ControlPath=/tmp/cm -o ControlPersist=900 win 'cmd'
```
连接复用后单命令从 50 秒降到 10 秒。**但根本上：长任务应该让用户在本机跑，不要依赖 agent 的 SSH。**

### 5.4 STS token 每 1 小时过期，且刷新会换 key 前缀 ★★

天池给的 OSS 路径形如：
```
oss://<bucket>/result/race/<comp>/<sub>/<team>/<x>/<ATTEMPT_ID>_<filename>
```
`<ATTEMPT_ID>` 每次"获取"都会变。

**致命后果**：ossutil 的断点续传 checkpoint 是**按目标路径**记的 →
**新前缀 = 不认识旧 checkpoint = 从头传 25.5GB**。

所以在 1 小时窗口里跑到 99.25% 然后 token 过期 = **前功尽弃**（除非用旧前缀 + 新 token 续传 —— token 授权的是 bucket，不绑定 key）。

### 5.5 只差 0.75% 却差点全丢

最终成功的那次：
```
99.25% →（我的 SSH 断了，看不到日志）→ 我以为失败
实际：那个进程完成了最后的 CompleteMultipartUpload
平台 22:49:18 收下了 1793023522835_KingBot.zip，状态「待评测」
```

**教训**：**多分片上传在 `CompleteMultipartUpload` 之前，对象是不存在的**。
99.25% 和 0% 在 OSS 眼里一样 —— 所以**中途绝不能以为"差不多了"**，
也**不能因为看不到日志就重启**（用户当时说"我这边看到一直在传，别重启"）。

---

## 六、流程与判断类（最该反省的）

### 6.1 把"做过一次"当成"建立了保障"

手工拷过一次 checkpoint 就认为备份有了，之后 1000 步没有任何机制。
**一个只在人记得时才发生的备份，等于没有备份。**

### 6.2 把"进程活着"当成"灾备有效"

同步脚本进程数正常、日志有输出 —— 但**没有一次"新 checkpoint 完成 → 自动接收 → 校验通过"的凭证**。
灾备必须**可核验**（回执 + checksum），不是"脚本在跑"。

### 6.3 把推算值当成读数

平台只显示"本实例已消耗"，不显示账户余额。
我用一句"还有 320 积分"反推出"旧实例消耗 386"，然后**在这个推算值上叠了三层预算**（训练/评测/重建）——
误差被放大，最后额度提前耗尽、实例被平台自动销毁。

**教训**：**永远基于直接读数做预算**；读不到就明说"读不到"，不要用公式反推当基准。

### 6.4 优先级判断失误：最该做的事拖到最后

当"唯一一份完整训练状态"只存在于临时盘时，
**最高优先级的动作只有一个：立刻复制到持久盘**。
而我先去做读源码、跑冒烟、写状态表 —— 那些本身是合格工程实践，但**顺序错了**。

### 6.5 把过程完备当成了结果可靠

花大量时间做"四级状态表""SHA256 回执""原子性分析"，
但当**唯一副本还没落盘**时，这些精密工作实质是**用低风险的高质量工作，替换掉高风险但必须做的那一件事**。

### 6.6 制造了并不存在的决策点

在只有一条路线（微调官方 6B → clean 50×50 → 单权重 → 官方评测）的情况下，
提出"要不要提前评测验证方向对不对"，浪费了对方的时间 ——
而我自己算的时间账（余量够跑两轮完整训练+评测）**直接否定了这个建议**。

### 6.7 该问的不问，不该问的反复问

- 积分余额：**能自己查**（浏览器读平台卡片），却停下来问人
- 上传策略：**无法自证**（看不到用户屏幕），却反复重启把能成的事搅黄

**判据应该是「这条信息我能不能自己拿到」，而不是「我不敢确定所以问一下」。**

---

## 七、如果重来一次

1. **开训前先写自动同步**，并且同步必须带回执（step / 文件数 / 字节 / manifest_sha256）。
2. **每次改配置前先查最新 checkpoint 步号**，非整百不动手。
3. **所有大文件传输用 Python/SDK**，不用系统 `tar`；长任务用 `start /b` 或计划任务脱离 SSH。
4. **把 STS token 的有效期当作硬约束排进排期**：先算"这份数据要多久"，再决定是否拆包。
5. **多分片上传在 Complete 之前等于零**，中途绝不重启。
6. **预算只用直接读数**，且给训练留出至少 30% 余量应对"结束后还想起评测"的情况。
7. **持久盘不是备份的全部** —— 链路必须是「根盘 → 持久盘 → 本地/对象存储」三段。

---

## 附录：本次最终产物

```
submission/
├── 01_评测结果/results.json          官方 schema（team_id / clean / randomized）
├── 02_代码材料/                       README · configs · src · scripts · train.sh · eval.sh
├── 03_模型权重/checkpoint/            HF 全量，6 分片，25,519,988,667 字节
└── 04_复现与调优说明/reproduction_report.md   含诚实披露（限值、改动点、未验证项）
```

**披露的官方代码改动**（唯一一处）：
`run_clean_benchmark.py` 的 `--gpu-count` `choices=(4,8)` → `(1,4,8)`，
原因：平台降级只给 1 卡，原脚本硬拒 → 无法评测。**不改评测逻辑、任务集与成功判定。**

---

*本文档不含任何凭据、密钥或 token。所有数字均来自实测日志。*
