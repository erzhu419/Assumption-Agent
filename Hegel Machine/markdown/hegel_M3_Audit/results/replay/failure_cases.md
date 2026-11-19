# V17.0 M3 Attribution Boundaries, Tradeoffs & Failure Cases (`failure_cases.md`)

## 1. 单纯一跳无歧义清洁目标（`T_SimpleClean`）上的诚实边界
- 在无跨树协议倒置且无枢纽词竞争的清洁目标上，`Flat_BM25_EqualMetadata` 与 `Hegel_M3_HippoPPR` 命中同一候选，PPR 不制造虚构物理测量优势，仅体现常数级查询计算差异；真正的结构增益集中在跨树关联升级簇（`T_Assoc`）与高频枢纽词稀释簇（`T_HubDilution`）。

## 2. 索引构建开销与目标流摊销边界
- `HippoRAGWormholeIndex` 每簇产生 `51.15` 逻辑计算建图成本（高于扁平表），以及源期近失反例检验与尺度探针成本；在单簇 `24` 道目标流内相对 `O_strong_AllValidatedPriors` 的端到端平衡点为 `N* = 7` 道目标。

## 3. 22 类机理库外函数（`OOS01`, `OOS02`）的诚实未知退出
- 在纯正弦波 `5 + sin(u)` 与二维调频振荡 `sin(a*b) + 0.5*cos(2a)` 共 `8` 个域外工作区上，前缀隔离确认 100% 拦截过拟合候选并诚实返回 `status='scoped_unknown'`，`false_confirmation = 0`。
