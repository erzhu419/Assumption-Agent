import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dynamic_replication", ROOT / "scripts/run_dynamic_research_replication.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_completed_code_and_old_hegel_cannot_evaluate_before_revised_hegel(tmp_path):
    protocol = json.loads((ROOT / "config/cognition_dynamic_research_replication.json").read_text())
    for name, arm in protocol["arms"].items():
        count = 4 if name == "hegel_revised" else 5
        study.base.save(tmp_path / "runs" / name / "151" / f"{arm['session']}_checkpoint.json",
                        {"session": {"tasks": [{}] * 5, "results": [{}] * count}})
    with pytest.raises(ValueError, match="all three learning sequences"):
        study.evaluate_arm(tmp_path, protocol, 151, "code")
    assert not list(tmp_path.rglob("evaluation_report.json"))
