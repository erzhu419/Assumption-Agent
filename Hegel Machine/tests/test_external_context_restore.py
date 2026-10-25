"""Restore external historical references while copying local writable models."""

from copy import deepcopy
import json

from hegel_machine.cognition import common_archive_inquiry, online_inquiry_llm
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from test_python_paid_score_reuse import ChoiceLab, scene


def test_restore_keeps_shared_external_execution_root_and_copies_local_context(
        tmp_path, monkeypatch):
    old, _ = scene(tmp_path / 'source')
    shared = tmp_path / 'historical_shared' / '_evaluated_model'
    shared.mkdir(parents=True)
    external = shared / 'model.py'
    external.write_text('def predict(row): return 3.\n')
    (shared / 'params.json').write_text('{"value": 3}')
    old.operator_results.append({'operator': 'research_choice', 'evaluations': [
        {'path': str(external)}, {'path': 'old/_evaluated_model/model.py'}]})
    old._prepare_program_state()
    previous = old.result('comparison_completed')
    copied, actual_copy = [], common_archive_inquiry._copy_execution_context

    def copy_context(source, *args):
        copied.append(source)
        assert source != shared
        return actual_copy(source, *args)

    monkeypatch.setattr(common_archive_inquiry, '_copy_execution_context', copy_context)
    monkeypatch.setattr(online_inquiry_llm, 'evaluate_python_model',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError('restoration must not fit or score')))
    resumed = CommonArchiveInquiryAgent(None, tmp_path / 'resumed', arm='hegel',
        source_task=False, config=old.config, knowledge=deepcopy(old.knowledge))
    laboratory = ChoiceLab()
    laboratory.costs = {'observation_cost': 0}
    restored = resumed.restore_practice_research(previous, old.workspace, laboratory,
                                                confirmation_seed=0)
    assert restored['new_observations'] == 0 and resumed.costs == old.costs
    assert resumed.operator_results[-1]['evaluations'][0]['path'] == str(external)
    assert old.workspace / 'old/_evaluated_model' in copied
    local = resumed.workspace / 'old/_evaluated_model/params.json'
    local.write_text('{"value": 7}')
    assert json.loads((old.workspace / 'old/_evaluated_model/params.json').read_text()) == {
        'intercept': 0}
    assert external.read_text() == 'def predict(row): return 3.\n'
    assert (shared / 'params.json').read_text() == '{"value": 3}'
