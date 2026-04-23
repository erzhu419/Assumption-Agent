"""Experiment 3: Meta-loop — distilling a meta-wisdom about the gate.

The agent examines the pattern of KEEP vs REVERT decisions in its own
validation_log_parallel.json and is asked to propose a single
meta-wisdom about its own A/B gate: what is it systematically
accepting or refusing, and what modification (if any) would improve it?

The output is a meta-wisdom candidate. Unlike the object-level wisdoms
that live in the library, this one targets the *orchestrator itself*,
and is stored separately as metalibrary candidates — it does not
automatically change the gate. Paper uses the existence of a coherent
proposal as evidence that one additional level of reflection is
reachable without new scaffolding.
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from gpt5_client import GPT5Client
from llm_client import parse_json_from_llm


AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_META = AUTO_DIR / "exp3_meta_wisdom.json"


META_PROMPT = """你是一个自反省 agent。下面是你过去 validation 里所有候选 wisdom 的 A/B 结果。阈值是 +10pp（wr_ext>=0.60 才 KEEP）。

## 所有候选及结果
{rows}

## 你的任务
从这些 pass/fail 的分布里识别**一个系统性的 pattern**——不是关于具体 wisdom 内容，而是关于 **gate 自己的行为**：

1. 在什么**类型的候选**上，gate 倾向接受 / 拒绝？
2. wr 分布的形状告诉你 gate 有什么**系统偏差**？
3. 阈值 +10pp 在这个样本量（n=50）下是**过紧、恰当、过松**？举证。

基于观察，**提出一条 meta-wisdom**——关于你自己 gate 行为的一条 orientation：

- aphorism: 关于 gate 行为的 aphorism (≤40 字，可自造或引用已有典故)
- source: 民间谚语 / 作者名 / "self-observed"
- when_applies: 这条 meta-wisdom 在什么 gate 决策时 fire
- proposed_mechanism: 如果你相信这条 meta-wisdom，下一轮应该怎么调整 gate（e.g. 调阈值、增加 test_n、改 judge 策略）
- falsifier: 如果下一轮的数据看到什么，你会放弃这条 meta-wisdom

## 输出 JSON（不要代码块）
{{
  "observed_patterns": [
    {{"pattern": "30-60字 描述 gate 的一个行为", "evidence": "用具体数字举证"}},
    ...
  ],
  "threshold_diagnosis": "过紧 / 恰当 / 过松 + 40-80字理由",
  "meta_wisdom": {{
    "aphorism": "...",
    "source": "...",
    "when_applies": "...",
    "proposed_mechanism": "...",
    "falsifier": "..."
  }}
}}
"""


def format_rows(validation_log):
    rows = []
    for entry in validation_log:
        for r in entry.get("results", []):
            ab = r["ab"]
            rows.append(
                f"  • [{r['decision']:7s}] wr={ab['wr_a']:.2f} "
                f"(ext {ab['wins_a']}/base {ab['wins_b']}/tie {ab['ties']}) "
                f"src={r.get('source','?')[:20]:20s}  {r['candidate'][:25]}"
            )
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    val_log = json.loads((AUTO_DIR / "validation_log_parallel.json").read_text())
    rows = format_rows(val_log)
    total = sum(len(e.get("results", [])) for e in val_log)
    n_keep = sum(1 for e in val_log for r in e.get("results", [])
                 if r["decision"] == "KEEP")
    print(f"Total validation rows: {total}  KEPT: {n_keep}")
    print(rows)

    client = GPT5Client()
    prompt = META_PROMPT.format(rows=rows)
    print(f"\n[meta] querying {client.model}...")
    resp = client.generate(prompt, max_tokens=1800, temperature=0.4)
    parsed = parse_json_from_llm(resp["text"])

    patterns = parsed.get("observed_patterns", [])
    diag = parsed.get("threshold_diagnosis", "")
    mw = parsed.get("meta_wisdom", {})

    print(f"\n=== Observed patterns ({len(patterns)}) ===")
    for p in patterns:
        print(f"  • {p.get('pattern','')}  [{p.get('evidence','')[:60]}]")
    print(f"\n=== Threshold diagnosis ===\n  {diag}")
    print(f"\n=== Meta-wisdom ===")
    for k, v in mw.items():
        print(f"  {k}: {v}")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator_model": client.model,
        "n_validation_rows": total,
        "n_kept": n_keep,
        "observed_patterns": patterns,
        "threshold_diagnosis": diag,
        "meta_wisdom": mw,
    }
    OUT_META.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_META.name}")


if __name__ == "__main__":
    main()
