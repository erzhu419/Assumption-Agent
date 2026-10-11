"""Reuse the recorded-decision runner for one parent return and one success.

Only the embedded/host reference recipe differs; all other cognition modules use
the previous cohort's actual after runtime. Replay is offline unless --live is
explicitly supplied for a fresh, fixed-prefix continuation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / 'artifacts/response_research_connections_live_20261010'
SAVED = ROOT / 'artifacts/response_research_connections_20261010'
ORIGINAL = ROOT / 'artifacts/response_current_public_view_live_20261010'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]


def main(variant, case, *, live=False, directory=None):
    import hegel_machine.cognition as cognition
    cognition.__path__[:0] = [str(SAVED / 'runtime_after'), str(SAVED / 'runtime_before')]
    recipe_path = HERE / ('world_reference_recalibration_' + variant + '.py')
    name = 'hegel_machine.cognition.world_reference_recalibration'
    spec = importlib.util.spec_from_file_location(name, recipe_path)
    recipe = importlib.util.module_from_spec(spec)
    sys.modules[name] = recipe
    spec.loader.exec_module(recipe)
    from scripts import run_response_matched_adoption as matched

    repetition, count = (1, 7) if case == 'failure' else (3, 4)
    recorded = PREVIOUS / f'live/repeat_{repetition:02d}/after/agent'
    prefix = ORIGINAL / 'changed_mechanism/repeat_03/hegel/agent'
    output = Path(directory).resolve() if directory else HERE / f'{variant}_{case}'
    inputs = (output.parent / (output.name + '_inputs') if directory
              else HERE / 'inputs' / f'{variant}_{case}')
    inputs.mkdir(parents=True, exist_ok=False)
    source = inputs / 'recorded' / 'agent'
    source.mkdir(parents=True)
    for decision in range(1, count + 1):
        directory = prefix if decision <= 3 else recorded
        index = decision if decision <= 3 else decision - 3
        (source / f'code_action_{decision:03d}.json').write_text(
            (directory / f'code_action_{index:03d}.json').read_text())
    # The runner compares original raw prefixes. Reference the existing local
    # record rather than copying tens of megabytes into another artifact.
    (source / 'laboratory_observations.json').symlink_to(recorded / 'laboratory_observations.json')
    baseline = inputs / 'cohort'
    baseline.mkdir()
    (baseline / 'protocol.json').write_text((ORIGINAL / 'protocol.json').read_text())
    bundle = json.loads((ORIGINAL / 'frozen_bundle.json').read_text())
    prepared = matched.prepare_bundle(bundle)
    (baseline / 'frozen_bundle.json').write_text(json.dumps(prepared, indent=2))
    runtime = inputs / ('runtime_' + variant)
    runtime.mkdir()
    inquiry = SAVED / 'runtime_after/world_llm_inquiry.py'
    (runtime / 'world_llm_inquiry.py').write_text(inquiry.read_text())
    protocol = {'original_cohort': str(baseline.relative_to(ROOT)),
        'cases': {case: {'source': str(source.parent.relative_to(ROOT)),
                        'arm': 'hegel', 'recorded_decisions': count}},
        'source_cohort': str(PREVIOUS.relative_to(ROOT)),
        'recipe': str(recipe_path.relative_to(ROOT)),
        'cognition_runtime': list(cognition.__path__),
        'source_repetition': repetition, 'live': live,
        'maximum_new_model_calls': 24 - count if live else 0,
        'scope': ('Original recorded actions and notes, then model-selected continuation under the original total limits.'
                  if live else 'Original recorded actions and notes only, until next unprovided decision or completion.')}
    (inputs / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    module_spec = importlib.util.spec_from_file_location('recorded_boundary_runner', PREVIOUS / 'replay.py')
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    runner.OUT = inputs
    summary = runner.run(variant, case, output, live=live)
    summary['source_repetition'] = repetition
    summary['runtime_recipe'] = str(recipe_path)
    summary['prepared_bundle_source'] = str(baseline / 'frozen_bundle.json')
    if case == 'positive':
        summary['original_observations'] = 222
        summary['preserved_full_success_trace'] = (
            summary['success'] and summary['original_observation_prefix_equal']
            and summary['costs']['environment_steps'] == 222)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary.get(key) for key in ('variant', 'diagnostic_case',
        'status', 'success', 'recorded_decisions', 'new_model_calls',
        'original_observation_prefix_equal', 'preserved_full_success_trace')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=('before', 'after'))
    parser.add_argument('case', choices=('failure', 'positive'))
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--directory', type=Path)
    arguments = parser.parse_args()
    main(arguments.variant, arguments.case, live=arguments.live, directory=arguments.directory)
