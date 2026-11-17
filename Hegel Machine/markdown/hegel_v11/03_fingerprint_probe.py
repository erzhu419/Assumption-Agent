#!/usr/bin/env python3
"""Check the reviewed prior fingerprint, not the full Hegel agent.

Default: execute an exact transcription of the small reviewed function.
--repo-root: extract that function's AST from a local checkout; require the pinned
file Git-blob SHA unless --allow-different-source is explicitly supplied.
No repository writes, network calls, model calls, or scientific benchmark runs.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Callable

HEAD = "abe15bead458739d1ef5b83e45e96140c75a4d34"
BLOB = "358f462b7455b965c6fb9afe670e9d62204cb8ea"
REL = Path("src/hegel_machine/cognition/premise_commitment_index.py")
REFERENCE = '''def _prior_state_fingerprint(prior: dict) -> str:
    payload = {
        "id": prior.get("source_concept_id"),
        "status": prior.get("status", "supported"),
        "ce_len": len(prior.get("counterexamples") or ()),
        "sup_len": len(prior.get("support_witness_rows") or ()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
'''

def load_function(root: Path | None, allow_other: bool) -> tuple[Callable, dict]:
    if root is None:
        source = REFERENCE
        origin = {
            "mode": "reviewed_function_transcription",
            "repository_checkout_executed": False,
            "full_agent_executed": False,
            "reviewed_commit": HEAD,
            "reviewed_file_blob": BLOB,
        }
    else:
        file = root / REL
        if not file.is_file():
            file = root / "Hegel Machine" / REL
        if not file.is_file():
            raise FileNotFoundError(f"Cannot find {REL} below {root}")
        raw = file.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != BLOB and not allow_other:
            raise ValueError(f"Source blob {blob} differs from reviewed {BLOB}. Use a pinned checkout.")
        tree = ast.parse(raw.decode("utf-8"), filename=str(file))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prior_state_fingerprint"]
        if len(nodes) != 1:
            raise ValueError("Expected exactly one top-level _prior_state_fingerprint")
        source = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
        origin = {
            "mode": "local_source_function_AST_only",
            "repository_checkout_executed": True,
            "full_agent_executed": False,
            "source_file": str(file),
            "source_blob": blob,
            "matches_reviewed_blob": blob == BLOB,
        }
    ns = {"hashlib": hashlib, "json": json}
    exec(compile(source, "<reviewed-fingerprint-function>", "exec"), ns)
    return ns["_prior_state_fingerprint"], origin

def run(fp: Callable, origin: dict) -> dict:
    base = {
        "source_concept_id": "prior:test", "status": "supported",
        "program": {"op": "mul", "value": 1.0},
        "research_program": {"coefficients": [1.0], "radius": 0.7},
        "scope": {"lower": 0.0, "upper": 1.0},
        "support_witness_rows": [{"features": {"q": 1.0}, "target": 1.0}],
        "counterexamples": [{"features": {"q": -1.0}, "target": 2.0}],
    }
    cases = {}
    for name in ("program", "coefficient", "radius", "scope", "support_content", "counterexample_content"):
        x = deepcopy(base)
        if name == "program": x["program"]["op"] = "add"
        elif name == "coefficient": x["research_program"]["coefficients"][0] = 9.0
        elif name == "radius": x["research_program"]["radius"] = 1.8
        elif name == "scope": x["scope"]["upper"] = 10.0
        elif name == "support_content": x["support_witness_rows"][0]["target"] = 100.0
        else: x["counterexamples"][0]["target"] = -100.0
        cases[name] = {"fingerprint_changed": fp(x) != fp(base), "fingerprint": fp(x)}
    changed_status = deepcopy(base); changed_status["status"] = "revoked"
    added_row = deepcopy(base); added_row["support_witness_rows"].append({"target": 2.0})
    return {
        "origin": origin,
        "base_fingerprint": fp(base),
        "semantic_mutations": cases,
        "undetected_semantic_mutations": sum(not x["fingerprint_changed"] for x in cases.values()),
        "controls": {
            "status_change_detected": fp(changed_status) != fp(base),
            "row_count_change_detected": fp(added_row) != fp(base),
        },
        "interpretation": "The function is blind to these content edits. Whether a production run encounters them requires a full call-path test; this is not an end-to-end stale-cache reproduction.",
        "model_calls": 0,
    }

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path)
    ap.add_argument("--allow-different-source", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    try:
        fp, origin = load_function(args.repo_root, args.allow_different_source)
        result = run(fp, origin)
    except (OSError, ValueError, SyntaxError) as exc:
        ap.exit(2, f"Error: {exc}\n")
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
