"""Exp 82 v2: 5 hypothesis proposers, one per `kind`.

Each proposer takes a wisdom (cid + aphorism + signal + unpacked) and returns
a typed Hypothesis with a kind-specific actionable `expr` payload. The
proposer uses an LLM (cheap-tier — claude-haiku-4-5) with a kind-specific
prompt that demands JSON output.

Output formats per kind:

  feature        → {"keywords_zh": [...], "keywords_en": [...], "regex": [...], "explain": "..."}
  constraint     → {"required_substrings": [...], "forbidden_substrings": [...],
                    "max_retries": int, "explain": "..."}
  decomposition  → {"steps": ["Step 1: ...", "Step 2: ...", ...], "explain": "..."}
  verification   → {"instruction": "...", "explain": "..."}
  hp_change      → {"temperature": float, "top_p": float, "max_tokens": int, "explain": "..."}

Persistence: each generated Hypothesis is appended to hypotheses.jsonl via
Hypothesis.persist().
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap  # noqa: E402
from hypothesis import Hypothesis  # noqa: E402


PROPOSER_BASE = """You are an expert in turning Chinese problem-solving aphorisms into concrete machine-actionable structures for an LLM solver pipeline.

WISDOM:
  Aphorism : {aphorism}
  Source   : {source}
  Trigger signal (when this wisdom should fire): {signal}
  Unpacked explanation: {unpacked}

KIND TO PRODUCE: "{kind}"

KIND DEFINITION AND OUTPUT SCHEMA:
{schema}

Output JSON only. No markdown, no commentary. The JSON object MUST match the schema for kind={kind}.
"""


_SCHEMAS = {
    "feature": """\
A `feature` is a 0/1 detector that fires on a problem description when the
wisdom should apply. We will run this detector on all 50 holdout problems
and check if it correctly distinguishes "should fire" from "should not fire".

Schema:
{
  "keywords_zh": ["短词1", "短词2", ...],   // 4-10 Chinese keywords/phrases that, if any matches, suggest the wisdom applies. Use SHORT phrases (1-3 chars/words).
  "keywords_en": ["term1", ...],            // 0-5 English keywords (only if helpful)
  "regex": ["pattern1", ...],               // 0-3 regex patterns (advanced); empty list is fine
  "explain": "..."                          // 1 sentence why these signals match the wisdom's domain
}""",

    "constraint": """\
A `constraint` is a post-hoc check on the SOLVER'S OUTPUT. After the solver
answers, we check if the answer contains specific required content (or
avoids forbidden patterns). If the constraint fails, the solver retries
(up to max_retries times). Constraints encode "the answer must DO X" where X
is the wisdom's prescription.

Schema:
{
  "required_substrings": ["必含短语1", "或者短语2", ...],   // 2-5 short substrings; answer must contain AT LEAST ONE
  "forbidden_substrings": [],                                 // 0-3 short substrings the answer must NOT contain (usually empty)
  "max_retries": 2,                                           // int 1-3
  "explain": "..."                                            // 1 sentence why this constraint encodes the wisdom
}""",

    "decomposition": """\
A `decomposition` is a sequence of explicit step instructions injected as a
PROMPT TEMPLATE before the problem. The solver is told "follow these N steps
in order, then produce the final answer." Each step encodes a piece of the
wisdom's procedural advice.

Schema:
{
  "steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ..."     // 3-5 steps total; each step is one concrete action
  ],
  "explain": "..."    // 1 sentence why this decomposition is the wisdom's procedural form
}""",

    "verification": """\
A `verification` is a SINGLE post-answer self-check instruction. After the
solver produces an answer, we append "Now verify your answer by: <instruction>.
If you find a problem, correct your answer." in the SAME prompt, asking the
solver to refine its own answer in one extra round.

Schema:
{
  "instruction": "...",   // ONE sentence describing a concrete verification action (e.g., 'substitute a numeric example back into the formula')
  "explain": "..."        // 1 sentence why this verify step encodes the wisdom
}""",

    "hp_change": """\
An `hp_change` is a SOLVER HYPERPARAMETER override that we hypothesize matches
the wisdom's epistemic stance. For example, an "investigate carefully" wisdom
might map to temperature=0.0 (deterministic, no creative leaps); a "consider
many angles" wisdom might map to temperature=0.7 (more diverse exploration).

Schema:
{
  "temperature": 0.0,     // float 0.0-1.0
  "top_p": 0.95,          // float 0.5-1.0
  "max_tokens": 2000,     // int 500-4000
  "explain": "..."        // 1 sentence linking these HP values to the wisdom
}"""
}


_JSON_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _extract_json(text: str) -> dict:
    """Strip markdown fences then pull the first {...} block."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object found in:\n{text[:500]}")
    return json.loads(m.group(0))


def _validate_payload(kind: str, payload: dict) -> None:
    """Light schema check; raise ValueError on shape mismatch."""
    if kind == "feature":
        if not isinstance(payload.get("keywords_zh"), list):
            raise ValueError("feature.keywords_zh must be list")
    elif kind == "constraint":
        if not isinstance(payload.get("required_substrings"), list) or not payload["required_substrings"]:
            raise ValueError("constraint.required_substrings must be non-empty list")
        payload.setdefault("forbidden_substrings", [])
        payload.setdefault("max_retries", 2)
    elif kind == "decomposition":
        if not isinstance(payload.get("steps"), list) or len(payload["steps"]) < 2:
            raise ValueError("decomposition.steps must be list of >= 2 strings")
    elif kind == "verification":
        if not payload.get("instruction"):
            raise ValueError("verification.instruction is required")
    elif kind == "hp_change":
        for k in ("temperature", "top_p", "max_tokens"):
            if k not in payload:
                raise ValueError(f"hp_change.{k} required")
    else:
        raise ValueError(f"unknown kind {kind!r}")


def propose(wisdom: dict, kind: str, llm_client=None,
            trigger_subset: list = None, outside_subset: list = None) -> Hypothesis:
    """Generate one Hypothesis of the given kind from the wisdom.

    `wisdom`: dict with keys cid, aphorism, source, signal, unpacked.
    `kind`:   one of feature / constraint / decomposition / verification / hp_change.
    """
    if kind not in _SCHEMAS:
        raise ValueError(f"unknown kind {kind!r}")
    if llm_client is None:
        # gpt_mini is the only cheap-tier model currently reachable on ruoli.dev
        # (gemini/claude_haiku returning invalid_grant 400 as of this run).
        llm_client = cheap("gpt_mini")
    prompt = PROPOSER_BASE.format(
        aphorism=wisdom.get("aphorism", ""),
        source=wisdom.get("source", ""),
        signal=wisdom.get("signal", ""),
        unpacked=wisdom.get("unpacked", "")[:1500],
        kind=kind,
        schema=_SCHEMAS[kind],
    )
    last_err = None
    for attempt in range(3):
        r = llm_client.generate(prompt, max_tokens=1500, temperature=0.4)
        try:
            payload = _extract_json(r["text"])
            _validate_payload(kind, payload)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"proposer failed after 3 attempts for {wisdom.get('cid')}/{kind}: {last_err}")

    explain = payload.get("explain", "(no explain)")
    h = Hypothesis(
        seed_cid=wisdom.get("cid", "unknown"),
        kind=kind,
        claim=f"[{kind}] {explain}",
        expr=payload,
        trigger_subset=list(trigger_subset or []),
        outside_subset=list(outside_subset or []),
    )
    return h


def propose_all_kinds(wisdom: dict, llm_client=None,
                       trigger_subset: list = None, outside_subset: list = None) -> list:
    """Generate one Hypothesis per kind for the given wisdom (5 total)."""
    out = []
    for kind in ("feature", "constraint", "decomposition", "verification", "hp_change"):
        h = propose(wisdom, kind, llm_client=llm_client,
                    trigger_subset=trigger_subset, outside_subset=outside_subset)
        out.append(h)
    return out


if __name__ == "__main__":
    # smoke test on WCAND10
    import json as _json
    M = _json.loads((Path(__file__).parent / "verdict_matrix.json").read_text(encoding="utf-8"))
    cands = {c["cid"]: c for c in M["candidates"]}
    wisdom = cands["WCAND10"]
    print(f"wisdom: {wisdom['aphorism']}")
    for kind in ("feature", "constraint", "decomposition", "verification", "hp_change"):
        try:
            h = propose(wisdom, kind)
            print(f"\n=== {kind} ===")
            print(f"  claim: {h.claim}")
            print(f"  expr: {_json.dumps(h.expr, ensure_ascii=False, indent=2)[:500]}")
        except Exception as e:
            print(f"\n=== {kind} === FAILED: {e}")
