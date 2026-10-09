# 高难任务选择

旧online公开四种响应函数、两个驱动坐标及有限标定组合，总共96个候选。Code目标29/30，已接近准确率天花板，继续增加这类随机种子不能有效扩大研究难度。旧任务保留为回归；此前26/30的Hegel结果保留。[旧协议](../artifacts/online_inquiry_luna_subscription_20261005_main_30/protocol.json)。

历史已有更难的短/长时序、联合机制变化、匿名对象和未知尺度研究。其中时序完整LLM小样本后来三臂均2/2；冻结Python方法新目标24/24不等于目标时LLM已经可靠。复用现成执行器和方法库，不再重建同类实验室。[历史入口](hegel_machine_reuse_hub_20261006.md)。

| 新任务集 | 已选清单 | 测什么 |
|---|---|---|
| DiscoveryWorld Normal + Challenge，主测 | 全8主题；seed0开发16例，seed1–4评测64例 | 操作仪器、形成规律、主动干预、修订解释并完成任务 |
| DiscoveryBench真实原始数据，补充 | `meta_regression_raw` 50题/20假设组，`nls_raw` 7题/7组 | 数据准备、中间变量构造、统计分析与假设发现 |

DiscoveryWorld八主题为组合化学、考古定年、植物营养、反应堆、陌生语言、疾病调查、蛋白质组学及火箭科学。使用全部主题和官方支持的种子0–4；开发/评测划分由本项目制定。[官方任务与动作额度](https://github.com/allenai/discoveryworld/blob/main/agents/README.md)、[完整清单与协议](../config/discoveryworld_hard_selection_20261007.json)。

DiscoveryBench选择两主题的全部test query，不按Agent表现选题。官方论文说明raw版本增加数据准备及中间变量推导难度；它没有统一的real-hard标签。[官方论文Appendix D.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/0d70af566e69f1dfb687791ecf955e28-Paper-Conference.pdf)、[57题清单](../config/discoverybench_hard_selection_20261007.json)。

## 开发起始额度与对照

| 额度 | 旧online实际配置 | DiscoveryWorld新开发配置 | DiscoveryBench新开发配置 |
|---|---:|---:|---:|
| 模型调用 | 16 | 2,000 | 128 |
| 环境交互 | 24行观测 | 2,000动作 | 静态数据分析 |
| 单次模型请求超时 | 180秒 | 900秒 | 900秒 |
| 单题研究墙钟上限 | 未单列 | 8小时 | 4小时 |
| 累计Python执行 | 60秒 | 7,200秒 | 7,200秒 |
| 累计reported token硬限 | 300,000 | 不设；记录实际及缓存用量 | 不设；记录实际及缓存用量 |

这些是待开发集校准的宽松起点，冻结正式实验前再确定。调用、动作、墙钟与Python时间分别计数；达到任务目标即可结束。模型调用额度不保证可执行同样数量的环境动作。DiscoveryWorld原论文Normal/Challenge上限1,000动作，另外报告1,000步成绩与2,000步扩展成绩，不混合。

主比较采用相同模型、公开工具、初始档案及资源条件的强普通Code/ReAct和Hegel。普通Code可以保存和执行自己的程序；共享初始档案只能来自开发数据。正式任务逐题重置，每臂3次执行，共384次，按主题、难度及种子配对；不是384种独立科学问题，也不直接充当跨任务迁移实验。沿用此前授权的`gpt-6-luna`作为请求型号，现已通过ChatGPT订阅CLI启动开发试跑。

## 状态与实施顺序

DiscoveryWorld公开接口已接通，官方运行源码与必需资产存于本机缓存（含Git约44MB）。Reactor Lab Normal/Challenge seed0均通过真实环境动作检查；5项针对性测试验证子答案改变父程序后续动作、反证与依赖变化使方法停用、官方成功不依赖manifest、评价信息不进入模型请求。复用旧CodexCodeProvider、InquiryLLMAgent执行器和select_actions协议；旧数值predict/MSE结算不适用于该环境，已替换为官方终局评分。

首批[四个运行](../artifacts/discoveryworld_pilot_20261007_v2/protocol.json)已停止（Normal两臂中断，Challenge尚未启动）：Reactor Lab两难度×Code archive/Hegel，两进程；[汇总](../artifacts/discoveryworld_pilot_20261007_v2/summary.json)与各臂active/result记录实际状态。两臂初始档案均为空，均可自由编写程序并使用问题/方法接口；Hegel指令强调依赖和子答案回传。首批检验操作接口、真实难度与研究采用，随后扩展全部16个固定开发场景，再冻结正式64题。结果若饱和或全失败，先区分科学难度、接口和资源截断，不用Hegel胜负筛选正式题目。DiscoveryBench仍仅有约482KiB元数据，没有下载CSV。

首轮试跑发现批次中间测量被后续移动消息盖住，仅留在证据文件；已补逐动作消息摘要并通过回归，两臂使用同样修复。原试跑作为[接口诊断](../artifacts/discoveryworld_pilot_20261007/RUN_STATUS.md)保留，v2又暴露证据指向/路径压缩/持续笔记断点，已停止保留，不把中断算科学失败。现在两臂共用修好的真实证据与笔记，Hegel接入旧method_decision当前支持摘要；21项定向测试通过。[24轮检查](../artifacts/discoveryworld_context_repair_20261007/RUN_STATUS.md)中两臂仍未保存可选笔记；现已统一每轮返回持久笔记，[同前缀各12轮续接](../artifacts/discoveryworld_persistent_notes_20261007/RUN_STATUS.md)24/24成功传递，但子研究采用仍0。长试跑与短检查均已结束，下一步解决真实研究采用，再决定完整开发运行。

运行入口：`python3 scripts/run_discoveryworld_pilot.py --output <new-directory> --tasks dw_04_normal_s0 dw_04_challenge_s0`；省略tasks运行全部16个开发场景，拒绝evaluation ID。官方runtime默认位于`~/.cache/hegel-machine/discoveryworld`，依赖pygame、numpy、Pillow、termcolor、pathfinding。评分检查点在各臂目录；模型仅在其agent子目录工作。

DiscoveryWorld任务成功与过程进展由模拟器评分；解释知识评分另需judge。适配器直接调用官方UI JSON生成器，保留一次性消息且不渲染PNG；两个真实环境检查均为零PNG。[官方API](https://github.com/allenai/discoveryworld/blob/main/discoveryworld/DiscoveryWorldAPI.py)。DiscoveryBench使用静态数据，官方HMS需LLM评判；原脚本指定`gpt-4-1106-preview`，接入时需统一可用评判模型并注明变更。[评判代码](https://github.com/allenai/discoverybench/blob/main/discovery_eval.py)。

## Limitations

DiscoveryWorld64题只有8主题，DiscoveryBench57题共享27个假设组和两个数据主题；按组报告，不能把变体当独立领域。2024年DiscoveryWorld低分只是选择背景，当前Luna的难度须实跑。任务完成不等于发现正确解释；泛化、递归组织增量与跨任务知识收益仍需各自对应的干预。
