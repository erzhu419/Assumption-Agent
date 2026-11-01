"""Continue paid interval and instrument research through retained references."""

from pathlib import Path

from scripts import review_paid_object_pool as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Probe
from test_paid_instrument_planning import (
    paid_scene, forbid_science_restart, frozen_instrument_probe_scene,
    frozen_probe_laboratory, forbid_refits_and_replanning,
)


def test_updated_interval_planning_uses_receipt_candidates_and_historical_libraries(tmp_path, monkeypatch):
    from hegel_machine.cognition import competing_object_experiments

    request, source, _, _, old_methods = paid_scene(tmp_path, monkeypatch, balance=7104)
    inherited = load(request)
    previous = load(source / 'registration.json')
    prior_result = load(source / 'receipt.json')
    candidates = [runner.ref(source / 'new_entries.json', index) for index in (0, 2)]
    binding_ref = runner.ref(tmp_path / 'retained_binding.json')
    binding = {name: Program('feature', name=control)
        for name, control in (('m_a', 'c_a'), ('m_b', 'c_b'))}
    save(Path(binding_ref['file']), {name: value.to_dict() for name, value in binding.items()})
    evidence_ref = runner.ref(tmp_path / 'paid_response_evidence.json')
    save(Path(evidence_ref['file']), {'status': 'distinguished'})
    save(source / 'registration.json', {**previous,
        **{key: inherited[key] for key in ('object_model_ref', 'instrument_models_ref', 'experiment_methods_ref')},
        'binding_ref': binding_ref, 'response_evidence_ref': evidence_ref,
        'prior_method_review': str(tmp_path / 'older_world_method_review'),
        'candidate_refs': [runner.ref(source / 'new_entries.json', 1)]})
    save(source / 'receipt.json', {**prior_result, 'status': 'paid_interval_updated',
        'candidate_refs': candidates})
    source_cost_refs = [inherited['source_costs_ref'], evidence_ref]
    supplied = tmp_path / 'updated_request.json'
    save(supplied, {'ledger_ref': inherited['ledger_ref'], 'source_cost_refs': source_cost_refs,
        'prior_method_review': str(tmp_path / 'retained_method_review')})
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'updated_response_plan'
    cfg = runner.register_response_plan(source, output, request=supplied)
    assert cfg['candidate_refs'] == candidates and cfg['available_compute'] == 7104
    assert cfg['ledger_ref'] == inherited['ledger_ref'] and cfg['source_cost_refs'] == source_cost_refs
    assert cfg['prior_method_review'] == str(tmp_path / 'retained_method_review')
    assert cfg['instrument_models_ref'] == inherited['instrument_models_ref']
    assert cfg['experiment_methods_ref'] == inherited['experiment_methods_ref']
    assert cfg['prior_plan_ref'] == evidence_ref and cfg['binding_ref'] == binding_ref
    assert cfg['compute_limit'] == 4008 and cfg['reserved_compute'] == 3096
    assert not any((source / name).exists() for name in
        ('plan.json', 'instrument_models.json', 'program_experiment_methods.json'))
    forbid_science_restart(monkeypatch)
    expected_entries = [runner.retained_entry(location, output, i) for i, location in enumerate(candidates)]
    calls = []
    probe = Probe('retained_method_prediction', ({'c_a': 1.6, 'c_b': -.5},))

    def plan(command, entries, budget, *, generation_budget, methods, **kwargs):
        assert [entry['candidate'].program for entry in entries] == [entry['candidate'].program for entry in expected_entries]
        assert generation_budget is not budget and generation_budget.limit == cfg['compute_limit']
        assert methods.records == old_methods and kwargs['binding'] == binding
        assert kwargs['questions'] == () and kwargs['serial'] == 2
        assert budget.spend(11)
        calls.append(entries)
        return {'probe': probe, 'forecast_status': 'predicted_distinction', 'predicted_separation': 2.1,
            'hypotheses': [{'candidate_id': entry['candidate'].candidate_id} for entry in entries],
            'predicted_fields': {'m_a': [1.6], 'm_b': [-.5]}}

    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe', plan)
    monkeypatch.setattr(competing_object_experiments, 'shorten_discriminating_probe',
        lambda plan, budget, **kwargs: plan)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_response_plan(output)
    assert result['status'] == 'response_distinguishing_experiment'
    assert result['compute_cost'] == 11 and result['remaining_compute'] == 7093
    assert result['source_cost_refs'] == source_cost_refs and result['ledger_ref'] == inherited['ledger_ref']
    assert result['new_world_fits'] == result['new_observations'] == result['model_calls'] == 0
    assert load(output / 'plan.json')['probe'] == {'probe_id': probe.probe_id, 'sequence': list(probe.sequence)}
    assert not load(output / 'plan.json')['acquired']
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_response_plan(output)['new_compute_cost'] == 0 and len(calls) == 1
    costs = tmp_path / 'probe_allowance.json'
    save(costs, {'cumulative_protocol_observations': 6, 'observation_limit': 8000})
    probe_request = tmp_path / 'execute_response_request.json'
    save(probe_request, {'source': str(output),
        'ledger_ref': runner.ref(output / 'receipt.json', 'remaining_compute'),
        'source_costs_ref': runner.ref(costs),
        'observation_limit_ref': runner.ref(costs, 'observation_limit')})
    monkeypatch.setattr(runner, 'ROOT', Path(__file__).resolve().parents[1])
    probe_cfg = runner.register_instrument_probe(probe_request, tmp_path / 'registered_response_probe')
    assert probe_cfg['prior_method_review'] == cfg['prior_method_review']
    assert probe_cfg['prior_method_review'] != load(source / 'registration.json')['prior_method_review']
    assert probe_cfg['instrument_models_ref'] == cfg['instrument_models_ref']
    assert probe_cfg['experiment_methods_ref'] == cfg['experiment_methods_ref']


def test_instrument_probe_restores_output_libraries_by_reference_without_local_copies(tmp_path, monkeypatch):
    from hegel_machine.cognition import (
        checkpoint_storage, competing_object_experiments, instrument_hypotheses, relation_uncertainty,
    )

    request, world, planned, probe, partial = frozen_instrument_probe_scene(tmp_path, monkeypatch)
    historical = tmp_path / 'retained_library_output'
    models_ref = runner.ref(historical / 'instrument_models.json')
    methods_ref = runner.ref(historical / 'program_experiment_methods.json')
    models = load(planned / 'instrument_models.json')
    methods = load(planned / 'program_experiment_methods.json')
    save(Path(models_ref['file']), models)
    save(Path(methods_ref['file']), methods)
    (planned / 'instrument_models.json').unlink()
    (planned / 'program_experiment_methods.json').unlink()
    receipt = load(planned / 'receipt.json')
    save(planned / 'receipt.json', {**receipt,
        'instrument_models_ref': models_ref, 'experiment_methods_ref': methods_ref})
    # Registration still names the old input libraries. The paid output wins.
    assert load(planned / 'registration.json')['instrument_models_ref'] != models_ref
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'reference_probe'
    cfg = runner.register_instrument_probe(request, output)
    assert cfg['instrument_models_ref'] == models_ref and cfg['experiment_methods_ref'] == methods_ref
    assert cfg['update_compute_upper'] == 21
    acquisitions = []
    frozen_probe_laboratory(monkeypatch, probe, acquisitions)
    forbid_refits_and_replanning(monkeypatch)
    checkpoint_bases = {}
    write_checkpoint = checkpoint_storage.write_json_checkpoint

    def write_with_reference_base(path, value, **kwargs):
        checkpoint_bases[Path(path).name] = Path(kwargs['base'])
        return write_checkpoint(path, value, **kwargs)

    monkeypatch.setattr(checkpoint_storage, 'write_json_checkpoint', write_with_reference_base)
    native = instrument_hypotheses.assess_instrument_hypotheses

    def assess(model, command, measured, pairs, budget, **kwargs):
        assert model == models[-1] and measured.context['object_model'] == partial
        assert len(pairs) == 3
        return native(model, command, measured, pairs, budget, **kwargs)

    class Questions:
        def __init__(self, *args, **kwargs):
            pass

        def questions(self, entries):
            return []

    def plan(command, entries, budget, *, methods, binding, **kwargs):
        assert methods.records[0]['method_id'] == 'old_boundary_method'
        assert methods.records[0]['support'][-1]['status'] == 'distinguished'
        assert binding['m_b'] == Program('feature', name='c_b')
        return {'probe': None, 'forecast_status': 'no_predicted_distinction', 'predicted_separation': 0.}

    monkeypatch.setattr(instrument_hypotheses, 'assess_instrument_hypotheses', assess)
    monkeypatch.setattr(relation_uncertainty, 'RelationIntervalResearch', Questions)
    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe', plan)
    monkeypatch.setattr(competing_object_experiments, 'shorten_discriminating_probe',
        lambda plan, budget, **kwargs: plan)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_instrument_probe(output)
    assert result['status'] == 'response_disagreement_unresolved'
    assert result['instrument_assessment_compute'] == 9 and result['pool_compute'] == 12
    assert result['compute_cost'] == 21 and acquisitions == [probe]
    assert result['new_world_fits'] == result['new_instrument_fits'] == 0
    assert checkpoint_bases == {'instrument_models.json': Path(models_ref['file']),
        'program_experiment_methods.json': Path(methods_ref['file'])}
    assert load(output / 'program_experiment_methods.json')[0]['support'][-1]['status'] == 'distinguished'
    assert all(path.read_bytes() == value for path, value in protected.items())
