# 失败任务进入原编辑实践循环（2026-10-08）

`prepare --practice-task-ids dw_04_challenge_s1 --frozen-practice`将已经打开的失败任务交给原WorldMethodLearner；首次实测失败在作者改稿前返回，原三个Normal源和续接案例继续作为保留条件。新任务不伪装成成功源档案，续跑保留案例及预算，发布包标记开发任务。相关23项测试通过。

真实Luna运行48调用、18实践、缓存输入1180928 token，Challenge八次均失败，未发布，最终恢复原稿。三个可执行改稿扩大对象遍历或跳过已有references，但收集references的代码始终只接受1/2，故仍误测晶体3。另有5次JSON解码错误、3次入口错误、错误operator和补丁断言失败各一次；没有服务超时。完整费用及逐次结果见`brief.json`。

## Limitations 与下一步

模型识别了问题，但没有完成角色修订。原四种起点的成功属于被保留的旧稿；后续公开角色绑定由宿主人工修复，另见[接入复测](../discoveryworld_public_role_integration_20261008/RUN_STATUS.md)。不把该修复计作模型自主成功。
