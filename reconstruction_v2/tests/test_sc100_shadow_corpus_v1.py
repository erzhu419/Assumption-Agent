from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1] / "reference" / "synthetic_sc100_shadow_v1"


def test_payload_and_self_hashes_are_exact() -> None:
    path = ROOT / "corpus_spec.json"
    specification = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in specification["payload_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    declared = specification["corpus_self_hash"]["value"]
    specification["corpus_self_hash"]["value"] = "0" * 64
    canonical = json.dumps(
        specification,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == declared


def test_case_order_uses_a_real_nul_separator() -> None:
    specification = json.loads((ROOT / "corpus_spec.json").read_text(encoding="utf-8"))
    case_ids = {
        case_id
        for cohort in specification["cohorts"].values()
        for case_id in cohort["case_ids"]
    }
    seed = specification["seed"].encode("utf-8")
    expected = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(
            seed + b"\0" + case_id.encode("utf-8")
        ).hexdigest(),
    )
    assert specification["ordering_algorithm"].endswith("NUL || UTF-8(case_id)")
    assert specification["case_order"] == expected


def test_cohort_sizes_and_gold_ids_match() -> None:
    specification = json.loads((ROOT / "corpus_spec.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (ROOT / "gold.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected_ids = set(specification["case_order"])
    assert len(rows) == 24
    assert {row["case_id"] for row in rows} == expected_ids
    assert specification["cohorts"]["required_positive"]["count"] == 12
    assert specification["cohorts"]["coverage_probe"]["count"] == 6
    assert specification["cohorts"]["true_negative"]["count"] == 6
