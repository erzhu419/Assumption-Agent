实际失败保留，修复后原测试另行通过。

- `migration_observation_funding.json`：旧概念短试与整轨迹噪声重复采购挤掉双条件实验；改名、尺度及逻辑变化后的独立测试MSE为5.794438。修复为保留已可采购的完整组合实验和确认资源。
- `continuation_confirmation_funding.json`：实测两处边界并共同拟合到约1e−20后，余40行不足硬设的24+24，未采用正确新答案。修复为缩短候选比较、保留完整24行独立确认。
- 联合拟合宽门槛的早期开发测试训练误差为AND 0.0783、OR 0.0296；交叉实验已辨明组合，但局部参数优化补偿了门槛错误。随后改为先实际二分收紧两门槛，再拟合完整响应；未扩大数值搜索循环。

初版开发续研变量绑定异常见 `../development/new_learning.log` 和相应 `failure.json`，冻结运行保留原样。

关联含噪窗口变化用例仍未完成，新旧版同为未决；实际报告见 `noisy_width_change_candidate.json.gz` 与 `noisy_width_change_baseline.json.gz`。本轮43项不同相关测试42通过，此既有失败保留。
