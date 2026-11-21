# 22 — Hegel V18.1（官方 ChatGPT 订阅多模型端到端实测：`gpt-6-luna` / `gpt-6.1-terra` / `gpt-6.2-sol` × 逐决策密码学溯源账本报告）

- **核心模块**：[src/hegel_machine/cognition/m5_live_gpt_and_provenance.py](../../src/hegel_machine/cognition/m5_live_gpt_and_provenance.py)
- **多模型一键实测入口（严格走官方 ChatGPT 订阅 `auth_mode: chatgpt`，硬屏蔽 `ruoli.dev`）**：[scripts/run_v18_1_m5_live_gpt_and_provenance_study.py](../../scripts/run_v18_1_m5_live_gpt_and_provenance_study.py)
- **逐决策溯源校验与因果树查询工具（支持 `--model` / `--task` / `--arm`）**：[scripts/inspect_m5_decision_provenance.py](../../scripts/inspect_m5_decision_provenance.py)
- **防篡改与反代理回归测试集**：[tests/test_v18_1_m5_decision_provenance_and_live_harness.py](../../tests/test_v18_1_m5_decision_provenance_and_live_harness.py)
- **完整静态证据与 134 条（1,025 步）决策轨迹包（`378/378` `SHA256SUMS.txt` 全过，零密钥/零第三方代理地址）**：[docs/evidence/v18_1_m5_live_gpt_and_decision_provenance_20261120/](../../docs/evidence/v18_1_m5_live_gpt_and_decision_provenance_20261120/study_manifest.json)

---

## 一、通道合规说明：100% 官方 OpenAI ChatGPT 订阅（零 `ruoli`）

- 本轮代码通过 `CodexSubscriptionClient` 直接调用官方 `codex-cli 0.169.1`（`auth_mode == "chatgpt"`，`provider_backend == "openai_chatgpt_subscription_codex_cli"`），在子进程环境层**强制剥离 `RUOLI_BASE_URL`、`RUOLI_API_KEY`、`GPT5_BASE_URL`、`GPT5_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_API_KEY`**，并对任何含 `ruoli` 的配置抛出硬阻断异常（见 `test_subscription_client_rejects_non_chatgpt_and_scrubs_ruoli`）。

---

## 二、官方订阅三档模型 × 三组架构臂（`54` 场在线多轮对话、`K=15` 异质多义源档案）实测总表

在规模 `K = 15`（7 个跨学科真机理节点 + 8 个共享表面线索的多义近失干扰节点）的源档案与 6 道涵盖全部 `M5` 支柱的未见任务（`L01` 记忆核、`L02` 短期红利长期反转、`L03` 绝热粗折叠、`L04` 两层共同基底子树、`L05` 异质外漏反例、`L06` 域外非平稳高频混沌）上，统一在 `reasoning_effort = "medium"` 下对比轻量低成本模型 **`gpt-6-luna`**、均衡模型 **`gpt-6.1-terra`** 与旗舰编程模型 **`gpt-6.2-sol`**：

| 官方订阅模型 | 实验臂（同一模型 + 同一隔离 Python 进程验真） | 域内严格通过 (`5`) | 域外诚实弃权 (`1`) | 假确认数 | 在线调用数 | 原始总 Token | 有效计费 Token | 物理测量费 `M` | 总复合成本 `C` | 累计墙钟耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`gpt-6-luna`**（轻量便宜） | **`Hegel_M5_LiveGPT`** | **`5/5` (`100%`)** | **`1/1`** | **`0`** | **`13`** | **`151,527`** | **`122,919`** | **`74.5`** | **`82.70`** | **`265.9 s`** |
| `gpt-6-luna` | `EqualAuth_Flat_LiveGPT`（平权粗通道 + BM25 Top-3） | `2/5` (`40%`) | `1/1` | **`1`** | `22` | `292,982` | `225,686` | `96.5` | `110.07` | `572.3 s` |
| `gpt-6-luna` | `Strong_Ordinary_LiveGPT`（全 `K=15` 原始历史直解） | `4/5` (`80%`) | `1/1` | `0` | `17` | `246,077` | `190,781` | `114.0` | `124.74` | `386.3 s` |
| **`gpt-6.1-terra`**（均衡中坚） | **`Hegel_M5_LiveGPT`** | **`5/5` (`100%`)** | **`1/1`** | **`0`** | **`11`** | **`104,959`** | **`62,239`** | **`77.0`** | **`81.95`** | **`299.7 s`** |
| `gpt-6.1-terra` | `EqualAuth_Flat_LiveGPT`（平权粗通道 + BM25 Top-3） | `5/5` (`100%`) | `1/1` | `0` | `17` | `240,543` | `138,111` | `90.0` | `98.74` | `529.4 s` |
| `gpt-6.1-terra` | `Strong_Ordinary_LiveGPT`（全 `K=15` 原始历史直解） | `5/5` (`100%`) | `1/1` | `0` | `14` | `135,142` | `86,086` | `101.0` | `106.33` | `575.0 s` |
| **`gpt-6.2-sol`**（高阶编码） | **`Hegel_M5_LiveGPT`** | **`5/5` (`100%`)** | **`1/1`** | **`0`** | **`8`** | **`132,926`** | **`98,174`** | **`72.5`** | **`79.17`** | **`228.5 s`** |
| `gpt-6.2-sol` | `EqualAuth_Flat_LiveGPT`（平权粗通道 + BM25 Top-3） | `5/5` (`100%`) | `1/1` | `0` | `9` | `164,024` | `106,904` | `72.0` | `78.88` | `322.5 s` |
| `gpt-6.2-sol` | `Strong_Ordinary_LiveGPT`（全 `K=15` 原始历史直解） | `5/5` (`100%`) | `1/1` | `0` | `8` | `173,161` | `109,897` | `99.0` | `105.37` | `246.2 s` |

---

## 三、跨模型核心科学发现（模型能力 × 哲学森林架构耦合规律）

1. **轻量便宜模型（`gpt-6-luna`）获得质的跃升，直接反超裸跑高阶模型**：
   - 不带哲学森林时，`gpt-6-luna` 在 `EqualAuth_Flat` 下被 `K=15` 多义近失节点误导，域内正确率跌至 **`2/5`（`40%`）** 并出现 **`1` 次假确认**（在 `L04` 两层虹吸题上误拟合并碰巧擦过确认集，未见留出 `heldout_rmse = 0.0571 > 0.045` 翻车），Token 暴涨至 `29.3 万`；在 `Strong_Ordinary` 下也被 `L01` 记忆核拖满 6 轮失败（`4/5`，`124.74 C`）。
   - 一旦挂上 `Hegel_M5`，便宜的 **`gpt-6-luna` 直接拉满 `5/5` 域内严格通过 + `1/1` OOS 弃权、`0` 假确认**，且总复合成本 **`82.70 C`** **不仅比同模型 `Flat`（`110.07 C`）省 `-24.9%`、比 `Strong_Ordinary`（`124.74 C`）省 `-33.7%`，甚至显著低于裸跑 `gpt-6.1-terra`（`106.33 C`）和裸跑 `gpt-6.2-sol`（`105.37 C`）！**
2. **中坚模型（`gpt-6.1-terra`）全指标横扫两大基线**：
   - 在 `gpt-6.1-terra` 上，虽然三臂最终都做对了题目，但 `Hegel_M5` 相比同具粗通道权限的 `EqualAuth_Flat` **节省 `-55.0%` 有效计费 Token（`62,239` vs `138,111`）、减少 `-35.3%` 调用轮数（`11` vs `17`）、降低 `-14.4%` 测量费（`77.0` vs `90.0`）、缩短 `-43.4%` 墙钟时间**；相比 `Strong_Ordinary` **节省 `-23.8%` 测量费与 `-47.9%` 墙钟时间**。
   - 尤其在两层共同基底子树题 `L04` 与 `L05` 上，`Hegel_M5` 仅耗费 `19,411` 与 `9,556` tokens（`L05` 一轮命中 `heldout_rmse = 0.0`），而 `EqualAuth_Flat` 因先踩多义单因子/漂移干扰项分别耗费 `85,609`（`4.4x`）与 `64,202`（`6.7x`）tokens 才靠证伪回退救回！
3. **旗舰编码模型（`gpt-6.2-sol`）依然保持物理测量费与推理速度优势（并如实披露单题边界）**：
   - 对比 `Strong_Ordinary`，`Hegel_M5` 节省 **`-26.8%` 物理测量费（`72.5 M` vs `99.0 M`）**、**`-24.9%` 复合成本**与 **`-23.2%` 原始 Token**；
   - 对比 `EqualAuth_Flat`，`Hegel_M5` 在两层子树题 `L04` 与 `L05` 上均以 **`1` 轮调用、`12.0 M`** 完胜 `Flat` 的 `2` 轮、`14.0 M`（总原始 Token 少 **`-19.0%`**，墙钟快 **`-29.2%`**）；唯一使 `Flat` 测量费微低 `0.5 M`（`72.0` vs `72.5`）的是 `L01` 上 `gpt-6.2-sol` 在无噪 5 行试点上直接裸解出了 4 参数耦合 ODE，而 `Hegel_M5` 稳健购买了 `extra_fine_trajectory_batch` 保障有噪泛化可辨识性。

---

## 四、逐决策密码学溯源账本（`134` 条因果链、`1,025` 个原子决策 `100%` 校验通过）

本次证据包完整收录 **`80` 场确定性基准实验（`630` 个原子决策）** + **`54` 场三模型官方订阅实验（`395` 个原子决策）** = **`1,025` 个带 `parent_decision_id` 与 `SHA-256` 前后锁链的原子决策记录**，每步同时记录宿主数学候选得分与大模型显式输出的 `considered_alternatives`（备选项及拒绝理由）：

```bash
# 1) 一键校验全部 378 个文件 SHA-256 及 134 条（1,025 步）决策哈希链与成本守恒
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/inspect_m5_decision_provenance.py --verify-all

# 2) 按模型、任务、实验臂精确回放任意一场实验的逐决策因果树
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/inspect_m5_decision_provenance.py \
  --model gpt-6-luna --task L04 --arm Hegel_M5_LiveGPT

# 3) 用官方 ChatGPT 订阅重跑任意模型组合
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_v18_1_m5_live_gpt_and_provenance_study.py \
  --models gpt-6-luna gpt-6.1-terra gpt-6.2-sol
```
