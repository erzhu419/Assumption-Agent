"""
Clean ablation setup:
Generate strategies_canonical_orientation/ — 12 JSONs with IDENTICAL schema
to strategies_orientation/, but `attention_priors` field filled with Polya/
Popper canonical heuristic questions (Chinese translations).

This lets us test "canonical CONTENT vs paraphrase CONTENT" with ALL OTHER
variables held constant (same prompt pipeline, same layer count, same
direct-warning framing).
"""

import json
from pathlib import Path

CANONICAL_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies_canonical")
ORIENT_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies_orientation")
OUT_DIR = Path("/home/erzhu419/mine_code/Asumption Agent/phase zero/kb/strategies_canonical_orientation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Clean orient-schema using canonical questions as attention_priors.
# Keep trigger minimal / neutral so the comparison isolates content.
# Paraphrase version has rich scenario-triggers like "当感觉被细节淹没时启动".
# Canonical version gets a GENERIC trigger — since canonical questions themselves
# carry no scenario specifics. This is a faithful test.

NEUTRAL_TRIGGER = "保持这些原典问句在读问题时浮现，不是一次性执行，是持续觉知"


def main():
    canonical_files = sorted(CANONICAL_DIR.glob("S*.json"))
    orient_files = {f.name: f for f in ORIENT_DIR.glob("S*.json")}

    for f in canonical_files:
        d_canon = json.loads(f.read_text(encoding="utf-8"))
        sid = d_canon["id"]

        # Get matching orient-schema JSON as template
        if f.name not in orient_files:
            print(f"  [skip] {sid}: no orient template")
            continue
        d_orient_template = json.loads(orient_files[f.name].read_text(encoding="utf-8"))

        # Build the new JSON: orient schema, canonical content
        qs_zh = d_canon.get("heuristic_questions_zh", [])
        qs_en = d_canon.get("heuristic_questions_en", [])
        src = d_canon.get("source", {})

        # attention_priors = canonical Chinese questions, unchanged
        new_d = {
            "id": sid,
            "name": d_canon["name"],
            "aliases": d_canon.get("aliases", []),
            "category": d_canon.get("category", ""),
            "form": "orientation",  # SAME form tag as strategies_orientation/
            "source_references": d_canon.get("original_source_references", []),
            "description": {
                "one_sentence": d_orient_template["description"]["one_sentence"],
                # ^ reuse paraphrase one_sentence so SELECT sees similar-looking module
                # This keeps SELECT behavior as similar as possible to orient_hybrid
                "background": d_canon["description"].get("background", ""),
            },
            "trigger": NEUTRAL_TRIGGER,
            "attention_priors": qs_zh,  # <-- CANONICAL QUESTIONS HERE
            "original_wisdom": [
                {"source": f"{src.get('author','?')}, {src.get('work','?')} ({src.get('year','?')})",
                 "text": q_en}
                for q_en in qs_en
            ],
            "applicability_conditions": d_canon.get("applicability_conditions", {}),
            "metadata": {
                **(d_canon.get("metadata", {})),
                "content_source": "canonical_heuristic_questions",
                "schema_version": "orient_v1",
            },
        }

        out_path = OUT_DIR / f.name
        out_path.write_text(json.dumps(new_d, ensure_ascii=False, indent=2))
        print(f"  ✓ {sid}: {len(qs_zh)} canonical priors in orient schema")

    print(f"\nWrote {len(canonical_files)} files to {OUT_DIR.name}")


if __name__ == "__main__":
    main()
