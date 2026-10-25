"""Exercise the runner save routes without acquiring observations or models."""

import ast
from copy import deepcopy
import gzip
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_text_incrementally
from hegel_machine.cognition import checkpoint_storage as central_storage

WRITERS = {
    'prepare_pr3_source_successor': '_write',
    'run_candidate_retention_cycle': 'write_json',
    'run_candidate_retention_cycle_llm': 'write_json',
    'run_code_agent_fresh_comparison': 'save',
    'run_cognition_v3_research': 'write_json',
    'run_cognition_v4_research': 'write_json',
    'run_cognition_v5_research': 'write_json',
    'run_cognition_v6_research': 'write_json',
    'run_deployment_learning_sequence': 'write',
    'run_discoveryworld_pilot': 'write_json',
    'run_dynamic_code_comparison': 'save',
    'run_compound_object_regression': 'write',
    'run_compound_experiment_methods_study': 'write',
    'run_instrument_competition_study': 'save',
    'run_instrument_method_study': 'save',
    'run_instrument_relation_code_comparison': 'save',
    'run_legacy_session_llm': 'write_json',
    'run_measurement_revision_study': 'write_json',
    'run_measurement_transfer_study': 'write_json',
    'run_mixed_research_investment_study': 'save',
    'run_object_competition_study': 'write',
    'run_object_equivalence_study': 'write',
    'run_observation_action_study': 'save',
    'run_partial_object_study': 'write',
    'run_program_disagreement_study': 'write',
    'run_relation_interval_confirmation': 'save',
    'run_response_composition_transfer': 'write',
    'run_response_family_controls': 'write_json',
    'run_response_numeric_transfer': 'write',
    'run_sc_kg_joint_revision_study': 'write',
    'run_sc_kg_meta_prior_study': 'write_json',
    'run_sc_kg_prior_adoption_study': 'write',
    'run_shared_transfer_object_regression': 'write',
    'run_stateful_instrument_study': 'save',
    'run_transfer_object_regression': 'write',
    'run_unified_action_policy_study': 'save',
    'run_unified_composition_study': 'write',
    'run_unified_research_study': 'write',
    'run_unified_state_risk_study': 'write',
    'run_unified_structure_study': 'write',
    'run_online_inquiry_study': 'write_json',
}


def _function(name, function, **extra):
    tree = ast.parse((ROOT / 'scripts' / f'{name}.py').read_text())
    definition = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == function)
    namespace = {'Path': Path, 'json': json,
                 'write_text_incrementally': write_text_incrementally,
                 'write_json_checkpoint': central_storage.write_json_checkpoint, **extra}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), name, 'exec'), namespace)
    return namespace[function]


def test_final_status_and_accounting_do_not_shift_inherited_checkpoint_payload(tmp_path):
    from hegel_machine.cognition.checkpoint_storage import write_chunks_incrementally

    save = central_storage.write_json_checkpoint
    active, final = tmp_path / 'active.json', tmp_path / 'training.json'
    original = {'status': 'researching', 'paid_evidence': 'retained counterexample ' * 5000,
                'costs': {'compute_cost': 17}}
    save(active, original)
    completed = {**original, 'status': 'method_step_limit',
                 'continuation_cost': {'compute_cost': 3}}
    save(final, completed)
    assert json.loads(final.read_text()) == completed
    assert json.loads(active.read_text()) == original
    unchanged_prefix = os.path.commonprefix([active.read_bytes(), final.read_bytes()])
    assert len(unchanged_prefix) >= len(active.read_bytes()) - 4096


def test_first_resumed_checkpoint_reuses_declared_prior_result(tmp_path, monkeypatch):
    from hegel_machine.cognition.checkpoint_storage import write_chunks_incrementally

    prior, current = tmp_path / 'prior.json', tmp_path / 'active.json'
    prior.write_text(json.dumps({'old_cost': 272, 'paid_evidence': [1, 2, 3]}))
    selected = []
    def save_chunks(path, chunks, *, base=None):
        selected.append(base)
        write_chunks_incrementally(path, chunks, base=base)
    monkeypatch.setattr(central_storage, 'write_chunks_incrementally', save_chunks)
    save = central_storage.write_json_checkpoint
    payload = {'resume_origin': str(prior), 'old_cost': 272, 'paid_evidence': [1, 2, 3]}
    save(current, payload)
    assert selected == [prior]
    assert json.loads(current.read_text()) == payload
    assert json.loads(prior.read_text())['paid_evidence'] == [1, 2, 3]


@pytest.mark.parametrize('name,function', WRITERS.items())
def test_periodic_writers_preserve_full_state_and_use_shared_writer(tmp_path, name, function):
    calls = []

    def record(path, text, **options):
        calls.append(Path(path))
        return write_text_incrementally(path, text, **options)

    def record_json(path, value, **options):
        calls.append(Path(path))
        return central_storage.write_json_checkpoint(path, value, **options)

    save = _function(name, function, write_text_incrementally=record,
                     write_json_checkpoint=record_json)
    path = tmp_path / 'checkpoint.json'
    original = {'paid_records': [{'target': index / 10, 'features': {'u': index}}
                                 for index in range(32)],
                'failure': {'status': 'refuted', 'reason': '旧反例'},
                'parameters': [0., 1.], 'costs': {'compute': 17}}
    save(path, original)
    revised = deepcopy(original)
    revised['parameters'][1] = 2.
    revised['costs']['compute'] = 23
    save(path, revised)
    assert json.loads(path.read_text()) == revised
    assert calls == [path, path]
    assert not list(tmp_path.glob('*.tmp'))
    assert not list(tmp_path.glob('*.writing'))


@pytest.mark.parametrize('name', [
    'run_knowledge_lineage_study', 'run_dynamic_code_comparison',
    'run_instrument_method_study', 'run_mixed_research_investment_study',
])
def test_saving_before_runtime_selection_does_not_import_mainline_science(tmp_path, name):
    # Importing the package to reach its storage helper would load today's
    # controller before the runner selects the registered baseline runtime.
    code = '''import importlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
runner = importlib.import_module(sys.argv[2])
runner.save(pathlib.Path(sys.argv[3]) / 'state.json', {'costs': {'compute': 19}})
assert not any(name == 'hegel_machine' or name.startswith('hegel_machine.') for name in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code, str(ROOT / 'scripts'), name, str(tmp_path)],
                   check=True, capture_output=True, text=True)


def test_old_gzip_is_retained_and_new_plain_checkpoint_restores_latest_state(tmp_path):
    flow = importlib.import_module('run_knowledge_lineage_study')
    old = tmp_path / 'checkpoint.json.gz'
    original = {'costs': {'compute': 41}, 'failure': 'retained', 'knowledge': {'a': [1, 2]}}
    with gzip.open(old, 'wt') as stream:
        json.dump(original, stream)
    frozen_bytes = old.read_bytes()
    assert flow.checkpoint_path(old.with_suffix('')) == old
    assert flow.read(old) == original
    revised = {**original, 'costs': {'compute': 47}}
    flow.save(old, revised)
    assert old.read_bytes() == frozen_bytes
    assert flow.checkpoint_path(old) == old.with_suffix('')
    assert flow.read(old) == revised
    assert flow.read(old.with_suffix('')) == revised
    with gzip.open(old, 'rt') as stream:
        assert json.load(stream) == original


@pytest.mark.parametrize('name', [
    'run_composed_condition_study', 'run_condition_response_action_study',
    'run_observed_condition_decoding_study', 'run_conditional_discovery_study',
    'run_nonlinear_mechanism_revision', 'run_knowledge_update_revisits',
    'run_research_investment_attribution_study', 'run_knowledge_lineage_study',
])
def test_completed_historical_gzip_is_not_rerun(tmp_path, monkeypatch, name):
    runner = importlib.import_module(name)
    arm, root = 'candidate', 'r'
    folder = tmp_path / 'runs' / arm / root
    if name == 'run_knowledge_lineage_study':
        folder /= 'stage00'
    folder.mkdir(parents=True)
    checkpoint = folder / 'checkpoint.json.gz'
    original = {'study': {'learning_complete': True}, 'costs': {'compute_cost': 59}}
    with gzip.open(checkpoint, 'wt') as stream:
        json.dump(original, stream)
    saved = checkpoint.read_bytes()
    loader = 'runtime' if name == 'run_research_investment_attribution_study' else 'load_runtime'
    monkeypatch.setattr(runner, loader, lambda *_: SimpleNamespace())

    def unexpected_save(*args, **kwargs):
        raise AssertionError('a completed historical experiment must not be recomputed')

    monkeypatch.setattr(runner, 'save', unexpected_save)
    plan = {'arms': {arm: {'knowledge': 'empty' if name == 'run_knowledge_lineage_study'
                          else 'cumulative'}}, 'roots': {root: {'tasks': [{}]}}}
    runner.learn(tmp_path, plan, arm, root)
    assert checkpoint.read_bytes() == saved
    assert not checkpoint.with_suffix('').exists()


@pytest.mark.parametrize('name', [
    'run_knowledge_update_revisits', 'run_research_investment_attribution_study',
])
def test_each_retained_visit_uses_its_committed_checkpoint_as_base(tmp_path, monkeypatch, name):
    flow = importlib.import_module('run_knowledge_lineage_study')
    runner = importlib.import_module(name)
    calls = []

    def record(path, value, **options):
        calls.append((Path(path), options.get('base')))
        return flow.save(path, value, **options)

    monkeypatch.setattr(runner, 'save', record)
    state = {'study': {'steps': 1}, 'knowledge': {'old': 'preserved'},
             'costs': {'compute_cost': 73}, 'counterexamples': [False]}
    runner.persist(tmp_path, 'candidate', 'r', state)
    current = tmp_path / 'runs/candidate/r/checkpoint.json'
    visit = tmp_path / 'runs/candidate/r/visit01.json'
    assert calls == [(current, None), (visit, current)]
    assert flow.read(current) == flow.read(visit) == state
    flow.save(current, {**state, 'costs': {'compute_cost': 79}})
    assert flow.read(visit) == state


def test_final_exports_share_actual_active_or_training_snapshot(tmp_path, monkeypatch):
    runner = importlib.import_module('scripts.run_online_inquiry_llm')
    calls = []
    write = central_storage.write_chunks_incrementally

    def record(path, chunks, **options):
        calls.append((Path(path), options.get('base')))
        return write(path, chunks, **options)

    assert runner.write_json is central_storage.write_json_checkpoint
    monkeypatch.setattr(central_storage, 'write_chunks_incrementally', record)
    value = {'costs': {'compute_cost': 83}, 'status': 'failed', 'failure': {'error': 'retained'}}
    runner.write_json(tmp_path / 'active.json', value)
    runner.write_json(tmp_path / 'training.json', value)
    runner.write_json(tmp_path / 'result.json', value)
    assert calls == [(tmp_path / 'active.json', None),
                     (tmp_path / 'training.json', tmp_path / 'active.json'),
                     (tmp_path / 'result.json', tmp_path / 'training.json')]
    assert json.loads((tmp_path / 'result.json').read_text()) == value


@pytest.mark.parametrize('name', [
    'run_anonymous_method_learning', 'run_anonymous_method_revision',
    'run_native_method_compilation', 'run_astra_code_frozen_comparison',
])
def test_reference_export_reuses_paid_trace_before_author_construction(tmp_path, name):
    from hegel_machine.cognition.code_agent import _copy_model_file, archived_paid_trace_references
    tree = ast.parse((ROOT / 'scripts' / f'{name}.py').read_text())
    block = None
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for index, statement in enumerate(function.body):
            if (isinstance(statement, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'previous_reference'
                            for target in statement.targets)
                    and index + 2 < len(function.body)
                    and isinstance(function.body[index + 2], ast.For)):
                block = function.body[index - 1:index + 3]
                author = next(n.value for n in function.body[index + 3:]
                              if isinstance(n, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == 'author'
                                      for target in n.targets))
                assert any(k.arg == 'paid_trace_references' for k in author.keywords)
                break
        if block:
            break
    assert block is not None
    source, output = tmp_path / 'source', tmp_path / 'next'
    previous = source / 'reference_archive'
    trace = previous / 'archive/source_a/paid_trace.json'
    trace.parent.mkdir(parents=True)
    trace.write_text('{"paid_rows":[1,2],"failure":"retained"}')
    (previous / 'parameters.json').write_text('{"coefficient":2}')
    files = {'archive/source_a/paid_trace.json': trace.read_text(),
             'archive/source_b/paid_trace.json': trace.read_text(),
             'parameters.json': (previous / 'parameters.json').read_text()}
    namespace = {'Path': Path, 'source': source, 'output': output,
                 'plan': {'source_root': str(source)}, 'original': {'files': files},
                 'original_knowledge': {'files': files}, 'knowledge': {'files': files},
                 'archived_paid_trace_references': archived_paid_trace_references,
                 '_copy_model_file': _copy_model_file,
                 'write_text_incrementally': write_text_incrementally}
    exec(compile(ast.Module(body=block, type_ignores=[]), name, 'exec'), namespace)
    copied = output / 'reference_archive'
    for key in ('archive/source_a/paid_trace.json', 'archive/source_b/paid_trace.json'):
        assert (copied / key).is_symlink()
        assert (copied / key).resolve() == trace.resolve()
    (copied / 'parameters.json').write_text('{"coefficient":3}')
    assert (previous / 'parameters.json').read_text() == '{"coefficient":2}'


def test_continuation_progress_preserves_source_and_uses_incremental_writer(tmp_path):
    path = ROOT / 'scripts/run_method_continuation_study.py'
    tree = ast.parse(path.read_text())
    definition = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == 'save')
    args = SimpleNamespace(output=tmp_path / 'continuation.json', checkpoint=tmp_path / 'old.json')
    calls = []

    def record(path, text):
        calls.append(path)
        write_text_incrementally(path, text)

    namespace = {'json': json, 'args': args, 'write_text_incrementally': record}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    report = {'source': {}, 'costs': {'compute_cost': 97}, 'failures': ['prior failure']}
    namespace['save'](report)
    assert calls == [args.output]
    assert json.loads(args.output.read_text()) == {
        **report, 'source': {'checkpoint': str(args.checkpoint.resolve())}}


@pytest.mark.parametrize('name,target', [
    ('run_cognition_llm_pilot', 'artifact'), ('run_code_agent_demo', 'args.output'),
    ('run_discoveryworld_hypothesis_extension', 'state_path'),
])
def test_direct_loop_state_export_uses_incremental_writer(tmp_path, name, target):
    tree = ast.parse((ROOT / 'scripts' / f'{name}.py').read_text())
    calls = []
    output = tmp_path / 'result.json'
    value = {'prior_paid_records': [1, 2], 'failure': 'retained', 'costs': {'compute': 101}}

    def record(path, text):
        calls.append(path)
        write_text_incrementally(path, text)

    namespace = {'json': json, 'args': SimpleNamespace(output=output),
                 'artifact': output, 'state_path': output, 'payload': value,
                 'report': value, 'state': value, 'write_text_incrementally': record}
    statement = next(n for n in ast.walk(tree)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == 'write_text_incrementally'
        and ast.unparse(n.value.args[0]) == target)
    exec(compile(ast.Module(body=[statement], type_ignores=[]), name, 'exec'), namespace)
    assert calls == [output]
    assert json.loads(output.read_text()) == value
