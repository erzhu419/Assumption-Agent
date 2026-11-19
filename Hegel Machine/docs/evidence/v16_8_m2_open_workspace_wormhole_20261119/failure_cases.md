# M2 Failure Cases, Honest Ties & Non-Transitive Drift Analysis (`failure_cases.md`)

## 1. 强基线 `O_strong_CodeSession` 命中同维无偏先验时的诚实同分项
- 在未发生符号反转或未重置记忆的基准目标（如 `T_M2B1_battery_dual_transport`、`T_M2C1_carrier_macro_window_mean`）上，赋能了 `shared/` 代码继承与 `rebuilt_cache` 先验重建的 `O_strong_CodeSession` 能复用同元数合成模板，与 `Hegel_M2` 取得相近的测量开销（`T_M2C1` 测量费同为 `20.0`）。
- **`Hegel_M2` 的真实增量来源**：主要来自 ① **跨符号/跨偏置读出透镜（`ObservationMap`）对齐**（如楞次 `-1` 感应端口 $\leftrightarrow$ 勒夏特列 `+1` 平衡移动 $\to$ TCP 拥塞反压 `T_M2A1` 与光学猝灭传感器 `T_M2C3`）；② **近失例驱动的状态扩充与封顶分裂（`CEGAR` + `ALIGN`）**（`T_M2A2` 热沉余温记忆、`T_M2B2` 机房供电制冷双限封顶）；③ **微观峰值尺度误用拦截**（`T_M2C2`）。

## 2. 双模粘弹蠕变（`T_M2A3`）与平滑协同（`T_M2B3`）的零假确认回退
- 在双时间尺度蠕变 `T_M2A3` 与平滑乘积协同 `T_M2B3` 上，单模松弛与分段 `min` 虫洞均在子进程独立确认或训练残差阶段被诚实拒绝，触发零假确认 Clean Exit 转入开放 Python 组合合成（严格通过率 `40/40 = 100%`）。

## 3. 多跳近似虫洞链（`T_M2C4_two_hop_drift_non_transitive`）对 `Flat_EqualContent` 的实测优势
- 在两跳近似中继（$\varepsilon_1=0.015, \varepsilon_2=0.018$, 目标容差 `0.025`）上，扁平等价类 `Flat_EqualContent_CodeBank` 仅校验单跳误差 $\le 0.025$ 而多跑一轮无效子进程拟合，而 `Hegel_M2` 通过 `compose_wormhole_chain` 检出累计误差 `0.033 > 0.025` 直接阻断，实现 `Hegel_M2 - Flat_EqualContent_CodeBank = -1.7437`（4/4 簇全负）。
