# M4.4 审计证据复现

固定源提交：`b965f022b214f47a1341559f20041da6a224d66d`。对照 M4.3：`3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`。审计日期：2026-11-20。

本包包含综合报告、三组专项报告、有限控制脚本及实际结果、完整官方重放输出和逐字段核验结果。它不包含整个 Git 仓库。先在原仓库检出上述固定 SHA，按 `Hegel Machine/pyproject.toml` 安装依赖。环境版本见 `environment.json`。

## 一 设置目录

下列示例中的 `HEGEL_AUDIT_REPO` 指向 **Hegel Machine 子目录**，同时适配本包所有脚本。路径中有空格时保留引号。`HEGEL_AUDIT_PACKAGE` 指向解压后的本审计目录。为重新执行选择一个新的 `HEGEL_AUDIT_REPLAY`，避免覆盖已归档证据。

```bash
export HEGEL_AUDIT_REPO='/absolute/path/Assumption-Agent/Hegel Machine'
export HEGEL_AUDIT_PACKAGE='/absolute/path/audit_m44_b965f022b'
export HEGEL_AUDIT_REPLAY='/absolute/path/new_m44_replay'
export PYTHONPATH="$HEGEL_AUDIT_REPO/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
```

git SHA 校验应在检出的仓库执行 `git rev-parse HEAD`。所有脚本以调用者选择固定源版本为前提，不会更改 git checkout，也不调用在线模型。

## 二 只核验现有证据包

无需重新进行科学拟合即可验证包中的独立重放输出与提交结果：

```bash
python3 "$HEGEL_AUDIT_PACKAGE/verify_replay.py" \
  --committed "$HEGEL_AUDIT_REPO/docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120" \
  --replay "$HEGEL_AUDIT_PACKAGE/replay/evidence" \
  --out "$HEGEL_AUDIT_PACKAGE/verification_rechecked.json"

python3 "$HEGEL_AUDIT_PACKAGE/recompute_part2.py" \
  --evidence "$HEGEL_AUDIT_PACKAGE/replay/evidence" \
  --out "$HEGEL_AUDIT_PACKAGE/part2_recalculated_again.json"
```

`verify_replay.py` 检查两侧每条 SHA、文件集合、完整 JSON/JSONL 逐字段差异、receipt 唯一键、phase 账本加和、`C=M+0.05K`，并单独记录 model SHA 与数值/状态差异。它不会用浮点舍入或路径替换掩盖差异。报告里的回放一致性以 `replay_verification.json` 为准。

## 三 官方全量实验与相关测试

在 `Hegel Machine` 目录执行；runner 的 `--out-dir` 是本提交支持的正式参数。不要省略它，以免把新结果写回源码仓库的默认 evidence 目录。

```bash
cd "$HEGEL_AUDIT_REPO"
python3 scripts/run_v17_6_m4_4_closure_and_writeback_study.py \
  --out-dir "$HEGEL_AUDIT_REPLAY/evidence"

python3 -m pytest -q tests/test_v16_*.py tests/test_v17_*.py
```

测试范围为 11 个 V16/V17 相关文件，不是整个仓库全部测试。完整 runner 包含 Part1 闭合控制、Part2 四种子十一臂及四种子噪声 fork、Part3 四臂在线流。两者部分内部调用重叠，是官方测试及全量研究入口自身的结构。实际原始日志保留于 `replay/`。

## 四 有限独立控制

### 身份和角色消费者

```bash
export HEGEL_AUDIT_OUT="$HEGEL_AUDIT_REPLAY/identity"
python3 "$HEGEL_AUDIT_PACKAGE/identity/probe_role_consumer_boundaries.py"
python3 "$HEGEL_AUDIT_PACKAGE/identity/probe_generic_or_actions.py"
```

前一个脚本检查16个 artifact action 的实例正负锚、规范/匿名角色、并保留早期 Pi OR 探索。后者是最终有效的无尺度证书普通 Monod OR 控制，使用本提交中已经确认并保存的 gain6/gain12 实例。Pi 混合实例探索在 JSON 中有 `scope_limit`，不得用其费用变化替代最终有效控制。

### 索引、证书与归档

```bash
export HEGEL_AUDIT_OUT="$HEGEL_AUDIT_REPLAY/index_archive"
python3 "$HEGEL_AUDIT_PACKAGE/index_archive/run_index_archive_controls.py"
```

该脚本从提交快照恢复，不重新形成整片森林。检查三个索引在正式 node/edge/utility 变更后的刷新、无变更不重复收费、证书新建与原子重验、三张快照及历史节点。仅两个原始 T21 控制实际执行拟合/确认，用来验证错误粗边是否被通道消费者使用。

### 在线写回

```bash
export HEGEL_AUDIT_OUT="$HEGEL_AUDIT_REPLAY/online"
python3 "$HEGEL_AUDIT_PACKAGE/online/analyze_online_receipts.py"
python3 "$HEGEL_AUDIT_PACKAGE/online/probe_s01_failure.py"
python3 "$HEGEL_AUDIT_PACKAGE/online/probe_s01_scope.py"
```

S01 scope 控制使用前一个 failure 控制保存的 before/after 状态，按此顺序运行。failure 控制只替换第一事件的 evaluator 返回，在 S02 前停止；它验证提交分支的判断条件，不声称已提交 S01 实际失败。

`twin_scope_control.json` 对应的有限控制已执行，使用本轮官方 pytest 生成的真实 W6 产物。三个必需来源文件已保存到 `online/twin_scope_bundle/`，其原始 pytest 路径和逐文件 SHA 见 `pytest_w6_artifact_provenance.json`；这些是审计补存的来源，不属于提交官方 evidence。

```bash
export HEGEL_ONLINE_W6_ARTIFACT="$HEGEL_AUDIT_PACKAGE/online/twin_scope_bundle/persisted_authored_workspaces/online_writebacks/online_W6_michaelis_cap_1d/authored_method.json"
export HEGEL_AUDIT_OUT="$HEGEL_AUDIT_REPLAY/online"
python3 "$HEGEL_AUDIT_PACKAGE/online/probe_twin_scope.py"
```

该控制仅调用冻结预测，不重新发现。报告关于 twin 的主要事实依据是源码、绑定身份及真实80条 receipts，不能把对源/目标参数不同的零拟合控制当作官方会重新拟合的 S04 失败。

## 五 解释边界

测试通过与逐文件复现支持已经执行的目标和合同；本次没有声称重读并重跑181个历史版本、所有论文或整个仓库的全部科学实验。审计主要结论分别标明原始缺陷闭合、当前有限接口控制和仍需另行证明的研究命题。

费用是项目的代理单位，`M` 与 `K` 不是现金。小数差异要区分逐行舍入、浮点产物差异和真正动作变化。PPR 的0.4550与pooling的零差额、Part2的合法冷恢复与Part3未保存在线状态，分别解释，不能相互替代。
