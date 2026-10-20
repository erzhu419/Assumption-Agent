"""Restored submissions remain shared evidence; evaluating fits is isolated."""

import json
from pathlib import Path

from hegel_machine.cognition import common_archive_inquiry
from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.models import Observation
from test_paid_scene_method_repair import make_repair


def test_repeated_practices_link_old_submissions_and_isolate_direct_and_bound_fits(
        tmp_path, monkeypatch):
    author, checkpoint, _ = make_repair(tmp_path)
    original = checkpoint.parent / '_submitted_models/confirmation_0000'
    model = original / '_evaluated_model/fit_touch.py'
    model.write_text('from pathlib import Path\n'
        'def fit(rows): Path("fit_touch.txt").write_text("new fit")\n'
        'def predict(row): return 3.\n')
    copytree = common_archive_inquiry.shutil.copytree

    def no_historical_copy(source, *args, **kwargs):
        assert not Path(source).resolve().is_relative_to(original.resolve())
        return copytree(source, *args, **kwargs)

    monkeypatch.setattr(common_archive_inquiry.shutil, 'copytree', no_historical_copy)
    first, second = [author._practice('same_scene') for _ in range(2)]
    for trial in (first, second):
        workspace = author.workspace / trial['directory']
        root = workspace / '_submitted_models'
        assert root.is_dir() and not root.is_symlink()
        assert (root / 'confirmation_0000').is_symlink()
        assert (root / 'confirmation_0000').resolve() == original.resolve()
        assert trial['continuation_cost']['observation_cost'] == 0
    monkeypatch.setattr(common_archive_inquiry.shutil, 'copytree', copytree)

    workspace = author.workspace / second['directory']
    path = '_submitted_models/confirmation_0000/_evaluated_model/fit_touch.py'
    assert evaluate_python_model(workspace, path, [Observation({'u': 0.}, 3.)])['mse'] == 0
    (workspace / 'parent.py').write_text('def bind_models(models):\n'
        '    global child\n    child=models["old"]\n'
        'def predict(row): return child(row)+1.\n')
    (workspace / 'model_bindings.json').write_text(json.dumps({'parent.py': {'old': path}}))
    score = evaluate_python_model(workspace, 'parent.py', [Observation({'u': 0.}, 4.)])
    assert score['returncode'] == 0 and score['mse'] == 0, score
    assert not (original / '_evaluated_model/fit_touch.txt').exists()
    assert not (workspace / 'fit_touch.txt').exists()
    # A later submission uses a new local directory, never the old evidence root.
    new = workspace / '_submitted_models/confirmation_0001/_evaluated_model'
    new.mkdir(parents=True)
    (new / 'model.py').write_text('def predict(row): return 4.\n')
    assert not (original.parent / 'confirmation_0001').exists()
