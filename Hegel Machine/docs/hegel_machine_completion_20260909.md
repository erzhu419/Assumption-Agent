# 0.26：模拟器、结构发现与跨任务代码研究

2026-09-09，0.26.0。三项剩余功能已进入代码和公共入口，并完成实际运行；十类功能均已有有界执行闭环。

- **模拟器自身修订。** 普通控制器从实际查询学习可执行的均值校正和条件方差，以后续轨迹比较、采用或撤销，保存后继续影响实验选择。[构造运行](../artifacts/cognition_simulator_revision_20260909.json)使用 51 行观测、700 计算，修订后实际改变查询。模拟器保存在对应工作区。
- **结构发现与应用。** 补齐有限投影、模块、动态、拓扑和响应次序研究；模型支持反例扩维、局部替换、慢变量表示、连接变动及稀疏条件例外。边界内仍有分歧时返回未知和具体实验。[15 个来源／复用／破坏案例](../artifacts/hegel_machine_v026_mechanisms_20260909/summary.json)全部完成，最高新观测 MSE 为 3.20e-18；研究使用 464 行观测、3488 测量、85571 计算，额外评价另计。
- **跨任务普通代码 Agent。** `CodeResearchSession` 保存代理自主写出的共享 Python、经验、实际观测与总预算；`CodeComparisonSession` 用同一装置任务驱动两臂，全部学习停止后评分。[八题实际比较](../artifacts/code_session_v026_summary_20260909.json)已全部取得可执行模型，7 个后续任务均在成功执行中导入了继承的共享代码。原 400000 token 协议得到 4 个已 finish 模型、2 个未 finish 访问和 2 个未访问任务，实际使用 413332 tokens；保留该结果后，按[单列扩展协议](../artifacts/code_session_completion_extension_protocol_20260909.md)恢复已成功评价的候选并完成剩余任务，累计 38 次调用、633842 tokens。

[24 阶段整机](../artifacts/hegel_machine_v026_complete_20260909_summary.json)已完成并接入八题代码比较，[600 项完整认知回归](../artifacts/cognition_v026_release_20260909.log)通过，覆盖修订后的低秩结构再次使用、预算耗尽时保留已成功执行的代码候选。运行入口是 `cognition frontier`、`cognition simulator`、`cognition mechanism`、`cognition agenda` 和 `cognition code-session`；结构研究可加入动态议程，代码比较可跨进程恢复。

八题平均 MSE：普通代码 Agent **0.02467**，黑格尔机 **0.06812**。双方各有两题明显占优，其余四题均接近零误差。代码 Agent 使用 264 行观测、1032 测量、408 评分计算，另有 5.16 秒 Python 和 1399.72 秒模型调用；黑格尔机使用 608 行观测、2348 测量、331547 搜索计算，研究用时 10.96 秒。下一步解决黑格尔机在平方潜在状态装置上遗漏有效测量的问题：训练内验证误差已经偏高，仪器重选仍未发生。

核心实现：[实验模拟器](../src/hegel_machine/cognition/experiment_simulator.py)、[结构发现](../src/hegel_machine/cognition/mechanism_discovery.py)、[响应与例外](../src/hegel_machine/cognition/response_profile.py)、[普通代码持续研究](../src/hegel_machine/cognition/code_session.py)。具体状态见[十项目标表](hegel_machine_advice_goal_status_20260909.md)。

## Limitations

结构和模拟器结果来自有限向量、标量及噪声装置；结构复用例使用同一机制的新观测。响应跳变仍保留约 7.63e-6 宽的未知区间。八题代码比较是四种装置各两个实例，扩展完成不代表原预算成功；代码评分计算、任意 Python 秒数和模型 tokens 与认知搜索计算不能直接等同。当前整体误差尚未领先，开放世界能力与稳定长期净收益仍未达成。
