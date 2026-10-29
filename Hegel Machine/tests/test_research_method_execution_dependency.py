"""A formed operation must not be silently ignored by its old executor."""

from hegel_machine.cognition.research_programs import export_research_selection_program
from test_method_startup_witness_binding import selection_plan


def test_binding_proposal_rejects_actual_old_adapter_contract_before_execution(tmp_path):
    old = tmp_path / 'old_ordinary.py'
    old.write_text("def select_actions(*args): return {'old': True}\n"
        "def select_research_segment(*args, **kwargs): return {'silently_ignored_binding': True}\n")
    plan = selection_plan()
    compiled = {}
    exec(export_research_selection_program(plan), compiled)
    result = compiled['select_actions']({}, [], {'ordinary_method_path': str(old)})
    assert result['stop']['request']['reason'] == 'ordinary_parent_binding_adapter_required'
    assert result['stop']['request']['ordinary_method_path'] == str(old)
    # The old unbound method remains executable with its original dependency.
    plan['when_false'] = {}
    exec(export_research_selection_program(plan), compiled)
    assert compiled['select_actions']({}, [], {'ordinary_method_path': str(old)}) == {
        'silently_ignored_binding': True}
