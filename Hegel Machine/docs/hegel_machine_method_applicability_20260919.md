# 研究方法适用条件与经验迁移

0.94落实最新[GPT6建议](../markdown/GPT6_advice.md)的认知动作选择：按实际可准备的修订项与已付历史区分经验，执行复用准备结果，收益由新响应确认决定。

同时修复零权重旧父项阻断修订的问题，原定义保留；默认动态仪器回归的观测144→64、测量168→0。相关测试24/24通过。

入口沿用[七阶段会话](../config/cognition_observation_action_session.json)，比较使用[研究脚本](../scripts/run_observation_applicability_study.py)。

四组新序列、每组七任务、三臂共84任务全部完成学习后统一终测；三臂均28/28通过、4/4完整序列。学习支出如下，终测费用三臂相同且另计：

| 版本 | 观测 | 测量 | 记账计算 | 查询 | 学习秒数 |
|---|---:|---:|---:|---:|---:|
| 0.93 | 1680 | 1620 | 236385 | 136 | 297.18 |
| 0.94 | 1680 | 1620 | 232263 | 136 | 322.92 |
| 0.94关闭策略 | 1952 | 1716 | 249976 | 144 | 312.96 |

新版较0.93计算少1.74%，观测、测量持平，耗时多8.66%；旧版在这些新样本中原本也选择正确。新版6次后继选择消费条件内经验，其中2次优先执行已确认方向。[完整结果](../artifacts/observation_applicability_study_20260919/main/final_digest.json) · [逐阶段选择](../artifacts/observation_applicability_study_20260919/main/applicability_digest.json)。

已知开发序列三臂均7/7，误迁移修复后测量849→513，两个变化阶段均恢复零测量修订。[开发结果](../artifacts/observation_applicability_study_20260919/development/final_digest.json)与[修补前结果](../artifacts/observation_applicability_study_20260919/development_v1/final_digest.json)分别保留。

## Limitations 与下一步

目前只识别既有有限历史修订方法的结构与数据前提，尚未自主生成新的方法适用谓词。局部修复有效，本批新序列未显示相对0.93的稳定净收益。关闭策略对照关闭整个研究策略；本批未新增普通Code对照。下一主线转向对象任务的共同研究选择，随后扩展风险任务。
