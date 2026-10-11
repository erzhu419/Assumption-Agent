"""Reuse recorded decisions with only the reopened assignment varied."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONNECTIONS = ROOT / 'artifacts/response_research_connections_live_20261010'
SAVED = ROOT / 'artifacts/response_research_connections_20261010'
REOPEN = ROOT / 'artifacts/response_research_reopen_20261010'
ORIGINAL = ROOT / 'artifacts/response_current_public_view_live_20261010'
COMMON = HERE / 'common_runtime'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_imports():
    # This directory contains the previous cohort's archived common sources,
    # complete cognition base and six-module after overlay. No live src fallback.
    if not (COMMON / 'src/hegel_machine/cognition/world_reference_response.py').exists():
        raise RuntimeError('Missing common frozen runtime')
    live_paths = {str(ROOT), str(ROOT / 'src'), str(ROOT / 'tests'), ''}
    sys.path[:] = [str(COMMON / 'src'), str(COMMON / 'tests'), str(COMMON)] + [
        path for path in sys.path if path not in live_paths]


def prepare(variant, case, output):
    frozen_imports()
    # Select before importing the matched bundle helper: its embedded controller
    # and reference_research_question must come from this same recipe.
    import hegel_machine.cognition
    recipe = HERE / ('world_reference_recalibration_' + variant + '.py')
    inquiry = HERE / ('world_llm_inquiry_' + variant + '.py')
    load('hegel_machine.cognition.world_reference_recalibration', recipe)
    from scripts import run_response_matched_adoption as matched
    inputs = output.parent / (output.name + '_inputs')
    inputs.mkdir(parents=True, exist_ok=False)
    source = inputs / 'recorded/agent'
    source.mkdir(parents=True)
    count = 7 if case == 'failure' else 20
    for decision in range(1, count + 1):
        if decision <= 3:
            directory = ORIGINAL / 'changed_mechanism/repeat_03/hegel/agent'
            index = decision
        elif decision <= 7:
            directory = CONNECTIONS / 'live/repeat_01/after/agent'
            index = decision - 3
        else:
            directory = REOPEN / 'live/repeat_03/after/agent'
            index = decision - 7
        (source / f'code_action_{decision:03d}.json').write_text(
            (directory / f'code_action_{index:03d}.json').read_text())
    reference = (CONNECTIONS / 'live/repeat_01/after/agent' if case == 'failure'
                 else REOPEN / 'live/repeat_03/after/agent')
    (source / 'laboratory_observations.json').symlink_to(reference / 'laboratory_observations.json')
    baseline = inputs / 'cohort'
    baseline.mkdir()
    (baseline / 'protocol.json').write_text((ORIGINAL / 'protocol.json').read_text())
    prepared = matched.prepare_bundle(json.loads((ORIGINAL / 'frozen_bundle.json').read_text()))
    (baseline / 'frozen_bundle.json').write_text(json.dumps(prepared, indent=2))
    runtime = inputs / ('runtime_' + variant)
    runtime.mkdir()
    (runtime / 'world_llm_inquiry.py').write_text(inquiry.read_text())
    protocol = {'original_cohort': str(baseline.relative_to(ROOT)),
        'cases': {case: {'source': str(source.parent.relative_to(ROOT)), 'arm': 'hegel',
                         'recorded_decisions': count}},
        'recipe': str(recipe.relative_to(ROOT)), 'inquiry': str(inquiry.relative_to(ROOT)),
        'common_runtime': str(COMMON.relative_to(ROOT)),
        'expected_boundary_observations': 223 if case == 'failure' else None,
        'expected_positive': {'model_calls': 20, 'observations': 275,
            'research_compute_units': 1708638, 'c5_confirmed_setting': 7906},
        'scope': 'Both arms retain parent-evidence reopening. Only the reopened caller assignment '
            'and its frame field differ. Recorded scientific choices, notes, original declarations '
            'and the other frozen runtime are shared.'}
    (inputs / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    return inputs


def main(variant, case, *, live=False, directory=None):
    if live and case != 'failure':
        raise ValueError('The complete positive trace is a zero-provider retention replay')
    output = Path(directory).resolve() if directory else HERE / f'{variant}_{case}'
    inputs = prepare(variant, case, output)
    runner = load('recorded_boundary_runner', CONNECTIONS / 'replay.py')
    # The reused runner adds live roots at import; remove these before any lazy
    # import. Its already imported scripts and fixture resolve to COMMON too.
    frozen_imports()
    runner.OUT = inputs
    summary = runner.run(variant, case, output, live=live)
    boundary = summary.get('next_decision') or {}
    summary['same_223_boundary'] = boundary.get('observations') == 223 if case == 'failure' else None
    summary['runtime_recipe'] = str(HERE / ('world_reference_recalibration_' + variant + '.py'))
    summary['prepared_bundle_source'] = str(inputs / 'cohort/frozen_bundle.json')
    if case == 'positive':
        point = any(trial['number'] == 5 and trial['physically_confirmed']
            and any(item['frequency'] == 7906 for item in trial['actual_positive_evidence'])
            for trial in summary.get('point_confirmations', []))
        summary['preserved_full_success_trace'] = bool(summary['success']
            and summary['original_observation_prefix_equal'] and summary['recorded_decisions'] == 20
            and summary['costs']['environment_steps'] == 275
            and summary['costs']['research_compute_units'] == 1708638 and point)
    sources = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
        if getattr(module, '__file__', None) and (
            name.startswith(('hegel_machine', 'scripts.')) or name.startswith('test_'))}
    (output / 'runtime_sources.json').write_text(json.dumps(sources, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary.get(key) for key in ('variant', 'diagnostic_case', 'status',
        'success', 'recorded_decisions', 'new_model_calls', 'same_223_boundary',
        'original_observation_prefix_equal', 'preserved_full_success_trace')}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=('before', 'after'))
    parser.add_argument('case', choices=('failure', 'positive'))
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--directory', type=Path)
    arguments = parser.parse_args()
    main(arguments.variant, arguments.case, live=arguments.live, directory=arguments.directory)
