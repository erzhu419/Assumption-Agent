# V17.3 / M4.1 Part 3 审计：可执行快慢动力学、Pi_s、区域证书与粗粒化边

固定提交：`99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`。

审计工作树：`/workspace/scratch/3f14ad2da0ed/repo_m41_99b1fdfeb/Hegel Machine`。

本报告只涉及 Part 3 及其直接依赖。原源码、原证据未修改；控制脚本、生成模型和结果全部在本审计目录。完整 study 与 33 项回归由主审负责，本分工未重复运行。

## 1. 结论

**这次的 RK4 微观动力学和 Schur 减阶确实实现了，比 M4 仅用两条解析读出加 Pi 名称有实质进步；粗粒化边也确实进入 PPR 并改变候选可达性。应保留这两项正面结论。**

仍有三个不同层次的问题：

1. **盒范围检查没有成为最终执行约束。** 只在召回成功路径检查第一个确认点，fallback 和最终预测没有相同检查。报告的 `out_of_box_query_rejected=True` 来自独立 helper 布尔测试，不能证明整条授权/执行链拒绝越界。真实调用控制可从盒外确认进入 fallback 后再次 `completed`。
2. **`72.6095 → 97.7765` 的数值可复现，但不能全部归为图检索不可替代的省测收益。** 其中测量差 24 来自 fallback 无条件多买 5 点、再多做 1 点确认；5 点中 2 点已经买过。仅复用已付 pilot、把确认点数与召回分支对齐，无边仍通过相同 heldout，成本降为 `73.7765`。原差额 `25.167` 的约 `95.36%` 随这项策略修改消失。
3. **步骤解释、计算计价及“轨迹投影”的表述仍比实际消费更强。** `fresh_points=2` 被校验但未执行；所谓“一次拟合”实际有两次 lens least-squares 加三次子进程模型 fit；窗口投影结果不进入在线通道、区域证书或预测，只在诊断函数中生成。

这些问题不否定原有盒内四点测试的质量，也不要求把有限经验闭合证书升级成全域数学定理。

## 2. 阅读范围与历史复用

已读 `AGENTS.md`、复用 hub 的相关历史条目，并保存了历史查询：

- `historical_fast_slow.txt`：多关键词同时查询无匹配，不据此认定历史算法不存在。
- `historical_ode.txt`：宽匹配目录，36 个版本项。
- `historical_pomdp.txt`：定位 `v10_world_structure_and_premise_audit` 路由，已有 POMDP、快慢/振荡动力学、跨尺度前提与代数证书能力。

直接复用和核对的现有能力包括：

- `m4_multiscale_renormalization.py:64–170` 的收费 coarse/micro 双通道、配对闭合探针和独立微观终测；新版本没有从零发明这些机制。
- 同文件 `1020–1073` 的 paid-prefix lens 标定，`_validate_and_interpret_m4_program_steps` 的步骤检查接口。
- `m4_compositional_discovery.py:53–113` 的独立执行与训练前缀隔离，`412–560` 的 AST 编译器及模型 fit。
- `code_agent.py:580–612, 735–753` 的子进程加载与训练视图。
- `hipporag_wormhole_index.py:226–242, 340–471` 的 coarse edge、PPR、尺度排序和候选输出。
- `wormhole_forest.py:219–348, 708–736` 的权限/ScaleMap gate 及 coarse edge 元数据。

新增主文件从 `85–898`（Part A 全部）逐路径阅读；相关测试为 `tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py:297–319`；工作总结为 `markdown/hegel_v16_5_multi_precision_plan/15_V17_3_M4_1_Audit_Closure_Executable_Pi_and_Unified_Chain_Report.md:37,55–70`。 runner 的 Part 3 调用只汇总该 suite 的输出。

## 3. 确实完成的改进

### 3.1 细尺度积分与粗尺度减阶来自同一套方程

新主文件 `85–142` 实际进行 RK4 时间步进，不是直接把答案解析式改名为 trajectory。细方程是：

\[
\dot x_s=-x_s/\tau_s+a_su/\tau_s+g_{sf}x_f,\qquad
\dot x_f=(-x_f+a_fu-g_{fs}x_s)/\tau_f.
\]

`145–186` 将快变量导数置零，使用

\[
x_f=a_fu-g_{fs}x_s,
\quad \gamma=1/\tau_s+g_{sf}g_{fs},
\quad b=a_s/\tau_s+g_{sf}a_f,
\]

再对 `dx_eff/dt = -gamma*x_eff + b*u` 做独立粗步长 RK4。读出使用相同 `c_slow/c_fast`，即 `(c_slow-c_fast*g_fs)*x_eff+c_fast*a_f*u`。由原细方程代入即可得到上述系数，代数自洽。

本审计使用同一线性系统的矩阵指数作为独立数值核验，`u ∈ {0.4,1.7,3.0}`、`T ∈ {0.08,0.12,2.2,4.0,5.2}` 共 15 点：

| 核验 | 最大绝对误差 |
|---|---:|
| 细 RK4 两个状态 vs 同系统矩阵指数 | `4.98691e-7` |
| 粗 RK4 慢状态 vs 减阶方程闭式解 | `3.18884e-8` |

因此，本分工没有发现“细/粗根本不是同一系统”或“RK4 只是注释”的问题。现有参数下细快慢模型可用两个衰减模表示，候选 `op_micro_two_timescale_ode_2d` 有明确物理对应。

### 3.2 尺度选择由实际已付差异驱动

`320–384` 会对相同点真实调用两个通道、收费、计算误差；`581–597` 使用得到的 `closure_verified` 决定 coarse 或 micro，并复用已购点。这里没有读取 `true_scale_regime`、heldout 误差或任务名来直接指定答案。 `sensor_lens` 只属于实验室生成观测，研究侧 lens 由已付 pilot 拟合。

短时 quench 的细尺度升级与 Gaussian 读出反例下拒绝粗闭合是有效控制。对于后者，细尺度候选未解释 Gaussian 读出而返回 `scoped_unknown`，原报告没有把该例算为成功求解。

### 3.3 粗粒化边确实改变了检索，不应延续上轮“边是摆设”的结论

图构建 `hipporag_wormhole_index.py:226–242` 把 coarse edge 加入转移矩阵，状态扩散在 `378–401` 执行。Part 3 的 coarse 节点没有任何 query signature 直接重合：

| 模式 | coarse edge 数 | 粗算子是否出现在召回结果 | 粗算子 PPR | 最终候选数 |
|---|---:|---|---:|---:|
| 原有边 | 1 | 是，`matched_signatures=[]` | `0.3253483` | 1 |
| 原无边 | 0 | 否，只余 micro/osc 直接命中节点 | 无 | 5 |

因此，这是一例真实的图边可达性干预。成立的窄结论是：在这个四节点图、给定 cue 与 fallback 策略下，边能触发更快的候选回取。更强的“全部 34.7% 都是必要图结构省下的研究代价”需作下文限定。

### 3.4 确认和终测没有向模型提供目标标签

Part 3 的候选拟合、确认和 heldout 分别调用 `evaluate_isolated_python_model`，训练前缀显式传入。底层 `code_agent.py:735–747` 把 eval 载荷投影为 features/history，训练标签只从 prefix 文件读取。`m4_compositional_discovery.py:85–100` 会替换 observations 文件而非把 eval rows 写进去。

在本分工读取的正常生成模型路径中，没有发现 heldout 标签参加候选选择或被直接喂给 fit 的情况。终测微观真值是在候选和确认之后生成并比较。

## 4. 主要缺陷：盒约束只检查局部分支

### 4.1 代码中的断点

- Lab 构造器 `237–246` 只保存 `declared_box` 与 `heldout_points`，不验证它们是否相容。
- `acquire_channel` 的 `300–306` 直接计算传入点，不检查 declared/verified box。
- 召回分支 `643–645` 只确认 `confirm_pts[:1]`。
- `649–652` 只检查上述**第一个确认点**是否位于 box。
- 通用授权函数不消费 `verified_input_box`，只在 `wormhole_forest.py:298–334` 检查 closure 标记、保留目标和丢弃变量的误差条目。
- 召回分支 `676` 被盒限制拒绝后，可以进入 `682–700` fallback；后者完全不做相同盒检查或 `authorize_abstraction_use`。
- `702` 最终预测/终测也不检查每个 query 的范围。
- suite 的 `848–851` 只对 helper 调用一次；`874–875` 将它的布尔结果作为“越界拒绝”证据返回。测试 `303–305` 再断言该布尔值，没有走真实 solver 的拒绝路径。

### 4.2 可复现控制与边界

本 Lab **没有单独的 `feature_bounds`/物理合法域字段**。`declared_box` 是 `u∈[0.4,3.0], T∈[2.2,5.2]`，证书直接复制它为 `verified_input_box`。以下 `T=1.2` 和 `T=0.12` 均在该已声明盒外；ODE 函数可以计算这些正时间，但它们不属于原盒内成功主张。

| 控制 | 只改变什么 | 实际行为 |
|---|---|---|
| `outside_confirm_fallback_near_box` | 两个确认点的 T 改为 1.2，原训练/证书/heldout 保持 | 召回确认误差 `2.05e-6`，被局部盒检查挡下；fallback 确认误差 `1.10e-6`，无最终授权检查便返回 `completed`、`confirmed_ok=True`、`strict_pass=True` |
| `outside_heldout_query` | 最终请求/heldout 换成 `(u=1.5,T=.12)`，原训练与确认保持 | 仍 `completed`、`confirmed_ok=True`；最终微观误差 `.275920`，`false_confirmation=True`；外部质量评估发现失败，但 scope guard 没有提前拒绝或升级 |
| `outside_confirm_fallback`，保留阴性结果 | 两个确认点 T 改为 .12 | fallback 的拟合外推误差 `.121035`，最终 `scoped_unknown`；不能把此例说成成功绕过。它是数值确认失败，并非统一范围拦截 |

对原已发布结果的正确解释是：**盒内主测试仍可复现、仍通过；报告“盒外查询 100% 被拒绝”的执行承诺没有得到实现。** 不是把越界测试伪装成原域内反例。

建议复用现有 scope/authorization 入口，把 box 绑定为执行契约，统一消费于召回、fallback、最终模型调用。对于越界请求可拒绝、重新做局部证书或升级 micro，具体策略由任务决定；不得在 recall 拒绝后由 fallback 绕过同一约束。测试需调用真实分支，而非仅测比较两个数的 helper。

## 5. 成本归因：边有效，但 24 的测量差来自可避免的 fallback 购测

### 5.1 原数据完全复现

| 路径 | 测量费 | 计价计算费 | 复合成本 | 候选数 | 微观 heldout RMSE |
|---|---:|---:|---:|---:|---:|
| 原有边 | 72 | 12.19 | 72.6095 | 1 | .008792 |
| 原无边 | 96 | 35.53 | 97.7765 | 5 | .008792 |
| 无边，仅复用已付 pilot、同为 1 点确认 | 72 | 35.53 | 73.7765 | 5 | .008792 |

第三行直接调用原嵌套 solver；只在 `acquire_channel` 的 `fallback_extra` 返回空增购、`fallback_confirm` 使用与召回相同的首个点。候选集合、顺序、训练已有数据、图删除、拟合器、终测和计算计价均保留；结果 `strict_pass=True`。

### 5.2 24 的分解

有边测量：`3*(4+12) + 5*4 + 1*4 = 72`。

无边测量：`3*(4+12) + 5*4 + 5*4 + 2*4 = 96`。

额外 `5*4` 来自新主文件 `684` 无条件 `pilot_pts[3:]`；但 `589` 已购 `pilot_pts[:5]`，所以索引 3、4 两点重复收费。额外 `1*4` 来自 `698` 取全部两个确认点，召回 `643` 只取一个。

原差额为 `97.7765-72.6095=25.167`，其中 `24/25.167≈95.36%` 是上述测量策略差。控制后余差为 `1.167`，约为有边成本的 `1.61%`。

计费本身没有凭空造数：这些额外调用确实发生、确实付了实验定价。因此最准确的批评是**归因范围过宽、fallback 策略未对齐**，而不是“结果伪造”或“删边根本没起作用”。应同时报告图召回机制效应和匹配实验/停止规则后的净成本效应。固定公开 fallback 顺序第一项本来就是 coarse spec；其所有候选被无条件全部拟合 (`687–696`)，也需要在公平停止规则中明确，而不应把候选数自动等同于必须完成的研究量。

## 6. 步骤与计算记录的剩余问题

### 6.1 `fresh_points` 校验后未执行

注册节点在 `538` 声明 `subprocess_isolated_confirm, fresh_points=2`。检索候选 `622–625` 只判断解释结果是否非空，`plan` 后续不使用；确认仍在 `643` 写死 `[:1]`。这属于“能拒绝非法 opcode，但合法步骤参数尚未支配执行”的集成缺口。主审的 B7 分工另有跨三个消费者的控制，本报告不重复扩大实验。

### 6.2 `evaluated_candidate_count` 不是实际 fit 次数

| 路径 | 记录的候选数 | lens 中 SciPy least-squares 次数 | 独立模型 fit 次数 | fine 积分次数 | coarse 积分次数 |
|---|---:|---:|---:|---:|---:|
| 原有边 | 1 | 2 | 3 | 7 | 9 |
| 原无边 | 5 | 16 | 7 | 7 | 15 |
| 无边复用 pilot、1 点确认 | 5 | 16 | 7 | 7 | 9 |

fine 次数含 3 次已付闭合探针及 4 次外部 heldout 计算；外部评测不计研究购测费是可接受的实验口径。每次隔离 evaluate 都重新加载模型并调用 fit，具体见 `code_agent.py:601–612`。lens 的联合参数/offset 标定也有优化，不能把它理解成只读一个均值。

`703` 的计算费是 `4 + 6*eval_count + query_compute_cost`，不消费这些事件的实际次数、迭代数或 RK4 步数。它可以作为明示的**抽象收费模型**，但报告“一次拟合”、实际计算节省、端到端净资源收益应另附事件计数与实测时间，不能用此代理量自动替代。此处不要求把所有 evaluator/基准真值计算重新算成研究预算，也不把缺少 wall-time 计费单独判为算法错误。

## 7. Pi_s 与区域证书的准确完成边界

### 7.1 可执行减阶已完成，轨迹窗口结果目前为诊断支路

`apply_executable_pi_s_projection` 在 `189–205` 可以接收 fine trajectory，计算尾窗平均投影与末态 slaving defect，这是可执行代码。但 `y_coarse` 在此前由独立粗 ODE 求出，和 fine trajectory 参数无数据依赖。

真实 `acquire_channel` 调用 `_eval_coarse_pi_s(clean)`，不传 fine trajectory；配对闭合证书也分别调用通道。控制中每条 solver 路径 `projections_with_fine_trajectory=0`。suite 的 `853–865` 才在离线诊断传入两条轨迹；返回报告只用了末态 defect，`window_projected_y` 没有被后续消费者读取。

所以应称为“真实细动力学 + 同模型可执行绝热减阶 + 轨迹投影诊断”，不要把它等同于“尾窗投影结果已经支配在线粗通道和区域授权”。若项目下一步需要后者，应接通现有函数输出，而非再造动力学算法。

### 7.2 区域探针实际是 3 点，文档中的四角/5 点标签不准确

`331–336` 是三对分位点 `(.08,.06),(.50,.45),(.94,.95)`；不是四角加中心。报告第 37 行“四角与内部中心”、函数 docstring 的 5-point 以及结果键 `Regional_5Pt_Box_Probe` 应勘误。handoff 已承认三点，本报告将此视为文档/标识债务，不再重复当成新发现。

有限经验采样本身并非缺陷。当前三点 `observed max = .009617`，在同一平滑无 bump 模型的盒角 `(u=3,T=2.2)`，粗细差为 `.039603`，仍低于当前容差 `.055`。这说明证书条目中的 `error_bound=.009617` 应明确为**采样到的最大差**，不能解释为已经证明的全盒上界；不影响本次已核验点通过。

无需再拿任意窄 spike 的有限采样不可保证性作为新问题：handoff 已明确这个限制。Gaussian 例是一个已配置正控制，恰在内点 `T=3.55` 被采中，支持该反例下多点探针比边界单点有效，未支持任意未知局部缺陷的完备检出。

### 7.3 Gaussian “微观共振”加入的是读出，而非状态动力学

`135–136` 在完成 RK4 后把 `local_micro_bump_fn(u,T)` 加到 `y_micro`，不改变 rhs、`x_slow` 或 `x_fast`。这是一个有效的观测/closure 失配反例；如果文档进一步说成真实系统产生的新动力学模、共振轨迹或投影发现新流形，应降格为“局部微观读出扰动”。当前拒绝粗闭合的证据仍成立。

### 7.4 这部分森林仍是手工先验，不是本轮发现出的 Pi 边

`465–548` 手工注册 4 个节点与 1 条 coarse edge；边上证书是固定 `defect_bound=.012`、`verified_regime='t >=18*tau_f'`。当前任务的实测证书确实用于通道选择和最终 recall 授权，但没有用于自动发现或修订这个 coarse edge。Part 3 因此是独立的重整化/检索机制验证，不能单独证明新 Pi_s 被作者发现、入林并在 Part 4 持续继承。

## 8. 建议优先级

| 优先级 | 最小必要动作 | 应有验收 |
|---|---|---|
| P1 | 将 verified box 接到所有授权/消费路径，fallback 复用同一检查 | 真正调用 solver/最终模型的越界请求应拒绝、再验证或切换细尺度；不能只断言 helper |
| P1（结论口径） | 将图可达性证据与匹配购测策略成本分开报告 | 保留原 72.61/97.78；并列报告同 pilot/同确认规则的对照，不把 24 自动归为必要图优势 |
| P2 | 消费有效 `program_steps` 参数，统一确认点数与可用预算 | `fresh_points` 改动在真实执行/账本产生相应行为，违规时不能进入成功状态 |
| P2 | 明示 compute 为代理收费，增加事件账本 | 候选、lens solve、model fit、观测与必要维护操作分别可核对 |
| P2/P3 | 准确界定 trajectory projection、经验误差与扰动位置 | 文档说明何种输出已进入执行；3 点/非四角/读出 bump 标签与源码一致 |

不建议因这些问题重写整个 ODE、PPR 或 AST 系统。已有能力足够，下一步主要是统一约束消费和校准归因。

## 9. 可复现材料

- `probe_pi_paths.py`：捕获原函数闭包、观测原调用，只有明确标注的 acquisition-policy 对照改变研究策略；不改源码。
- `results/probe_results.json`：六个控制摘要、15 点解析核验、原区域证书与盒外 API 调用记录。
- `results/original_with_edge.json`、`results/original_without_edge.json`：原结果逐调用复现。
- `results/without_edge_reuse_pilot_one_confirm.json`：匹配购测策略对照。
- `results/outside_confirm_fallback_near_box.json`：T=1.2 的真实 fallback 越界消费。
- `results/outside_heldout_query.json`：最终 T=.12 查询未被 scope guard 拦截，外部质量评估发现失败。
- `results/outside_confirm_fallback.json`：保留 T=.12 确认路径的数值失败结果，防止选择性报告。
- `probe_stdout.jsonl`：每次控制摘要。

运行命令：

```bash
python3 /workspace/scratch/3f14ad2da0ed/audit_m41_99b1fdfeb/pi/probe_pi_paths.py \
  --repo '/workspace/scratch/3f14ad2da0ed/repo_m41_99b1fdfeb/Hegel Machine' \
  --out /workspace/scratch/3f14ad2da0ed/audit_m41_99b1fdfeb/pi/results
```

所有结论针对固定提交及上述明确条件。未使用外部论文替代代码证据，数值推导由实现方程直接核验。
