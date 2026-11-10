# Qwen920F：构建一次，兼容节点复用

适用合入PR301及PR309的Qwen非MTP或MTP、单机920F TP16：DDR NUMA0～15各38个CPU，HBM NUMA16～31。本工具只管理部署，不改模型、算子和框架。完整验证范围见交付记录，不把历史性能当成本修订的实测成绩。

## 快速启动（单节点）

本入口启动一套单节点TP16服务，Prefill和Decode在同一服务内执行。无需Router或PD测试控制器。

```bash
TOOL=/绝对路径/sglang/scripts/cpu_kunpeng/qwen/deploy
DEPLOY=/绝对路径/deployments/qwen_20261106_mtp3_920f_tp16
mkdir -p "$DEPLOY"
cp "$TOOL/config.example.sh" "$DEPLOY/config.sh"
# 填写config.sh顶部的路径；MTP3设SGLANG_ENABLE_MTP=1、SGLANG_SPECULATIVE_NUM_STEPS=3。
# 新源码首次部署保留KERNEL_BUILD_RECEIPT为空，使用本次build的结果。
bash "$TOOL/run.sh" build "$DEPLOY/config.sh"
bash "$TOOL/run.sh" pack "$DEPLOY/config.sh"
bash "$TOOL/run.sh" start "$DEPLOY/config.sh"
```

出现READY后，在同一节点另开终端并设置相同TOOL/DEPLOY，再运行：

```bash
bash "$TOOL/run.sh" request "$DEPLOY/config.sh"
bash "$TOOL/run.sh" bench "$DEPLOY/config.sh"
bash "$TOOL/run.sh" status "$DEPLOY/config.sh"
bash "$TOOL/run.sh" stop "$DEPLOY/config.sh"
```

后续启动相同release只需start；源码或原生依赖更新时换一个DEPLOY_DIR并重新build/pack。start始终使用该目录的冻结release，不会自动切换到config中另一个源码目录。

本次接入保留上游PR301/PR309的模型实现。旧验证版的性能不代表本次上游合入版已完成原生构建或整模性能验证。

## 1. 配置与输出

工具位于SGLang的`scripts/cpu_kunpeng/qwen/deploy`。填写一份`config.sh`：源码、模型、Python、KuTACC/KuPL、BiSheng运行库、资源工具路径、端口、HBM及频率配置。`DEPLOY_DIR`使用版本/依赖/拓扑名称，不使用运行节点名。

`SGLANG_ENABLE_MTP=0`关闭MTP；设为`1`时使用`SGLANG_SPECULATIVE_NUM_STEPS`（默认3），固定topk=1、draft tokens=steps+1。启动传入NEXTN，框架归一为EAGLE。共享checkpoint可将`SPECULATIVE_DRAFT_MODEL_PATH`留空，需要独立draft权重时填写目录；实例状态和启动记录保存实际开关与步数，测试入口同时支持两种模式。

```text
DEPLOY_DIR/                         例如pr301_20261105_920f_tp16
├── config.sh                       可修改的默认配置
├── KERNEL_BUILD.json               本目录构建时生成的记录
├── PACKAGE.json                    pack状态
├── build/                          本目录构建时生成的CMake及kernel
├── release/                        冻结应用、kernel、打包时的工具
├── numa_images/rank0..rank15/       pack一次生成的原生库独立副本
├── instances/
│   ├── current/<主机>_<端口>.json   当前实例指针，不再共用LAST_RUN
│   ├── checks/<检查ID>/            verify/deps/check各自的报告
│   └── <时间_主机_端口_PID_ID>/     配置快照、STATE.json、资源记录、PID、测试JSON
└── logs/
    ├── driver/                     各命令的输出，包含主机、时间、PID
    ├── build/                      configure/build/install.log
    ├── pack/                       NUMA副本生成日志
    └── instances/<实例ID>/          launcher/rank/请求测试/调频等文本日志
```

不生成artifacts或重复压缩包。测试代码不放进SGLang。多节点共享同一个DEPLOY_DIR时，发布文件和副本只保留一套，各节点运行记录分开；每个节点仍须单独申请调度资源。

## 2. 首次从源码构建、打包

```bash
TOOL=/绝对路径/sglang/scripts/cpu_kunpeng/qwen/deploy
DEPLOY=/绝对路径/deployments/pr301_20261105_920f_tp16
mkdir -p "$DEPLOY"
cp "$TOOL/config.example.sh" "$DEPLOY/config.sh"
# 填写config.sh中的路径与参数。
bash "$TOOL/build.sh" "$DEPLOY/config.sh"
bash "$TOOL/pack.sh" "$DEPLOY/config.sh"
```

build使用CMake从源码生成sgl_kernel。pack校验构建记录、源码、SDK和DSO，冻结应用，并在相同920F拓扑的构建节点制作16份NUMA副本。start不会执行编译、pack或再次复制镜像。

有同版本、同依赖的完整构建记录时，可以在config中填写`KERNEL_BUILD_RECEIPT`，跳过build，只pack一次。已有release和numa_images不会被覆盖；算子/应用/依赖配方改变时使用新的版本目录。

## 3. 每次启动

在已分配的独占920F计算节点执行，保持原命令形式：

```bash
bash "$TOOL/run.sh" verify "$DEPLOY/config.sh"
bash "$TOOL/run.sh" start "$DEPLOY/config.sh"
```

start流程：

```text
取得本节点TP16管理锁、创建独立实例
  → 核验冻结应用/运行库/拓扑及现有副本
  → 保存本次配置与资源快照
  → 按配置准备大页、HBM/SHM预算及频率
  → 用已有NUMA副本启动16个rank
  → 短请求、选频/同NUMA换核、预热
  → READY并记录频率状态
  → stop后清理自有进程/SHM、恢复本次资源改动
```

副本里的hostname只表示制作来源，换节点不会因此要求重新pack。仍核验软件版本、依赖、Python路径和拓扑。镜像文件可复用不代表物理页永远本地：新节点仍需正确绑核、加载及核验内存放置。

共享文件系统下可直接复用；本版保留绝对路径和inode校验，不承诺把文件搬到任意新路径后免准备启动。不同机器没有共享路径时，应在目标环境准备一次副本并保持依赖匹配；不会因后续每次启动而再准备。

## 4. 请求、测试、状态、停止

同一计算节点另开终端，默认用“当前主机＋配置HTTP端口”定位实例：

```bash
bash "$TOOL/run.sh" status "$DEPLOY/config.sh"
bash "$TOOL/run.sh" request "$DEPLOY/config.sh"
bash "$TOOL/run.sh" bench "$DEPLOY/config.sh"
bash "$TOOL/run.sh" stop "$DEPLOY/config.sh"
```

也可以显式指定完整实例ID：

```bash
bash "$TOOL/run.sh" status "$DEPLOY/config.sh" --instance 实例ID
bash "$TOOL/run.sh" stop "$DEPLOY/config.sh" --instance 实例ID
```

共享文件系统下，从登录节点可通过显式ID查看状态或写入协作停止标记。request/bench要求在该实例所在节点执行，避免将另一节点的localhost误认作服务；跨节点发请求可使用HTTP客户端，指定实际节点地址。shared_tools的HTTP客户端继续可用。

同一个deployment同一主机不允许同时启动两个完整TP16管理进程；不同主机互不阻塞。不会回退到其他节点的LAST_RUN。异常杀死管理进程时，文件锁自动释放，但远程服务/资源是否已清理必须依据实例记录确认；不要仅凭锁释放就认定清理成功。

## 5. 资源配置沿用

- 每rank总HBM 4000MiB：SHM 384MiB、私有HBW池3616MiB。SHM包含在总预算内；prefill/decode当前保持同值，不相加两次。
- `HBM_HUGEPAGE_MODE=ensure|check`；`HBM_HUGEPAGE_MB_PER_NODE=auto`按私有池换算2MiB大页并留2页，当前1810页。保留原配置并只准备/恢复自有改动。
- `CPU_FREQ_MODE=apply|keep`，目标`CPU_FREQ_TARGET_MHZ=2000`、容差`CPU_FREQ_TOLERANCE_MHZ=20`。
- `CPU_FREQ_RESTORE_MHZ`按管理员确认的原标称值填写；无法可靠确定时不猜测恢复频点。
- `CPU_REMAP_SAME_NUMA=1`沿用原工具在同NUMA替换计算worker，不跨NUMA；不能达到要求时明确告警并继续服务。READY和频率合格分开记录。
- 调频、大页操作沿用授权的root_proxy；不新增权限机制。启动要求调度器已分配完整608CPU，不自行申请其他节点。

## 6. 更新部署工具，不重打已有应用包

只调整部署管理逻辑时，从已更新的SGLang工具目录执行相同run.sh命令即可：新工具读取DEPLOY_DIR中既有的冻结release和numa_images，不跳转回旧release中的旧run.sh。每个实例记录实际工具路径/修订及应用发布位置，以便追溯。

已有旧release里的run.sh保持原样；它不会自动升级，请使用SGLang中的新入口。需要带走完整的新工具版本时，在下一个新版本目录pack；不要手改旧release的完整性清单。本修订不修改原生库、不重新编译、不重建已有镜像。
