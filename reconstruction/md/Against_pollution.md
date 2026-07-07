问题成因：**Fast Mode 之前看似泛化，实际被多层污染干扰了判断。**

最早 BioFoundry 表现好，不完全说明流程强，而是因为系统里残留了 BioFoundry/enzyme/protein/DBTL 专用捷径。后来这些 hardcode 影响了 query、rank、guardrail 或 prompt prior，使其他领域被迫套用不适合自己的匹配逻辑。再叠加 OpenAlex/AGRIS 等 source pool 会召回大量 method-only、coauthor、aggregator、非 current PI 候选，自动 ledger 又曾把 weak/lookup_failed/empty official URL 当成质量信号，导致“看起来成功”的结果和严格联网审核差距很大。

分析结论：**当前主问题不是单一 query 参数，而是污染控制。**
清掉 BioFoundry hardcode 后，BioFoundry 没明显崩，Zhang Yifan 和 AGR-02 反而出现改善信号，说明通用 schema 路线是可行的。但要证明泛化，必须在更多 slots 上同时看质量和污染来源，而不是只看 final_count 或自动 object/bad rate。

解决方法：

1. **彻底禁用领域 hardcode**
   - 删除 BioFoundry 专用入口、固定 query、score boost/cap、legacy prompt variant。
   - 每次 Fast Mode 都跑 no-domain-hardcode gate。
   - gate 不通过则不能算 success。

2. **统一走 generic applicant schema**
   - 用申请材料抽取 object/method/domain/community/source hints。
   - HUM/SOC 允许 topic/community/source corpus/policy setting 作为 object-equivalent。
   - query/source/rank/evidence gate 都必须可追溯到 applicant schema。

3. **严格 cache + live 审计**
   - verified cache 可复用。
   - 新 PI / evidence changed 必须联网审。
   - historical provisional direct/strong 不可算提升。
   - lookup_failed / empty official URL 不可产生质量 verdict。

4. **污染来源落账**
   - selected query pollution
   - candidate pool pollution
   - official URL pollution
   - cache verdict pollution
   - advisor eligibility pollution
   - lookup/accounting pollution

5. **扩大到 12 slots 做 probe**
   - 不进 validation/holdout。
   - 每个 slot 同时输出质量表和污染表。
   - 判断哪些 domain 真改善、哪些只是污染减少、哪些仍卡在 official resolver/source seed。

一句话：**先把污染控制和审计口径固定住，再谈召回和 source lane 扩张；否则每轮实验都会把假阳性当成改进。**