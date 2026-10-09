"""Parallel terminal workers can read one another's updated checkpoints."""

import importlib.util
import json
from pathlib import Path


def test_checkpoint_stays_readable_during_another_workers_write(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "run_observation_action_study.py"
    spec = importlib.util.spec_from_file_location("observation_study", script)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    checkpoint = tmp_path / "checkpoint.json"
    study.save(checkpoint, {"results": ["learned"], "evaluation": []})
    reads = []

    def interleaved_write(path, text):
        with path.open("w") as stream:
            stream.write(text[:len(text) // 2])
            stream.flush()
            reads.append(json.loads(checkpoint.read_text()))
            stream.write(text[len(text) // 2:])

    monkeypatch.setattr(Path, "write_text", interleaved_write)
    updated = {"results": ["learned"], "evaluation": ["scored"]}
    study.save(checkpoint, updated)
    assert reads == [{"results": ["learned"], "evaluation": []}]
    assert json.loads(checkpoint.read_text()) == updated
