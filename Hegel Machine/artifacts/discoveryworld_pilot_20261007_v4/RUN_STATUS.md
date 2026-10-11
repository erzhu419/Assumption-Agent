# DiscoveryWorld v4：修复后开发试跑

原计划 Reactor Normal/Challenge seed0 × Code/Hegel 四项，两并发；现已停止，Normal两臂未完成，Challenge未启动。沿用订阅 gpt-6-luna medium、2000动作/调用、8小时、请求900秒，无累计token上限。缓存用量照常保存。protocol.json是实际配置；adapter_source_at_start保存本轮适配器和直接数值组件源码。

接回0.62的付费JSON错误反馈；修复USE入场/Exit后的公开对话菜单；两臂共享公开数值绑定→旧synthesize/choose_probe→区分实测→独立确认→父程序执行。模型仍须自行提出变量、提取公开读数及构造可执行实验。

验证：真实官方对话2项、错误恢复8项、数值研究9项、父子闭环8项通过，原相关程序测试通过。真实运行未完成数值绑定或父子研究闭环。

各运行active.json保留最后进展；summary.json记录本批停止状态。v3失败及停止记录原样保留。

## 操作前提诊断后停止

最后检查点见diagnosis_stop.json：Hegel147次调用、146动作，Code159次调用、226动作。Hegel仍仅根任务、0数值绑定/研究程序/子答案，普通Python反复读日志。第106次调用时尚未找到菜单；此后旧代码自发通过USE打开菜单，并在观测142、145读到参考频率3880.56、2239.92Hz。这是修复前的真实进展，不能归给后续改动。Code已使用菜单、测试gap×100猜想，并执行过1次检查线索程序；两臂均未完成任务。停止时在途调用用量可能未计入。

保留本批为真实采用负结果。新增[同一失败前缀续测](../discoveryworld_prerequisite_check_20261007/RUN_STATUS.md)从第106次调用、104条观测出发，两臂都不带入上述后期读数；它验证主机触发的操作前提研究，不替代完整任务比较。
