import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_summary_uses_object_baseline_and_retains_missing_sequence(tmp_path):
    protocol = json.loads((ROOT / "config/cognition_object_relation_revision.json").read_text())
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    seed = protocol["paired_seeds"][0]
    directory = tmp_path / "runs" / "hegel_v079" / str(seed)
    directory.mkdir(parents=True)
    costs = {"observation_cost": 12, "measurement_cost": 24, "compute_cost": 36}
    entry = {"task_position": 0, "research": {"costs": costs, "selected_channels": []},
             "actual_costs": {"queries": 1}}
    report = {"session": {"status": "running", "results": [entry], "costs": costs}}
    (directory / "hegel_checkpoint.json").write_text(json.dumps(report))

    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_object_relation_revision.py"),
                    "--output", str(tmp_path)], check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "summary.json").read_text())
    baseline = result["arms"]["hegel_v079"]
    assert baseline["expected_sequences"] == 6
    assert baseline["expected_tasks"] == 36
    assert baseline["visited_tasks"] == 1
    assert baseline["recorded_learning_costs"] == costs
    assert result["arms"]["code"]["recorded_learning_costs"]["observation_cost"] is None
    assert result["paired"]["hegel_revised_minus_hegel_v079"]["evaluated_pairs"] == 0
    assert result["paired"]["hegel_revised_minus_hegel_v079"]["observation_cost"]["pairs"] == 0
