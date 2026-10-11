#!/usr/bin/env python3
"""Recover complete paid export inputs using the unchanged frozen researcher."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from scripts import run_response_pending_band_revision as previous

OUTPUT = ROOT / 'artifacts/response_condition_retention_20261011'
CASES = (
    ('preservation', 'square_original__gain_higher__adapted_law'),
    ('preservation', 'square_original__gain_higher__original_law_return'),
    ('new_objects', 'square_original__gain_lower__adapted_law'),
)
read, write = previous.read, previous.write


def replay(cohort, case):
    source = previous.OUTPUT / cohort / 'after'
    output = OUTPUT / 'replays' / cohort
    protocol = read(source / 'protocol.json')
    selected = {name: protocol['cases'][name] for group, name in CASES if group == cohort}
    write(output / 'protocol.json', {**protocol, 'cases': selected,
                                   'planned_executions': len(selected)})
    config = selected[case]
    bundle = Path('sources') / config['source_id'] / 'numeric_bundle.json'
    write(output / bundle, read(source / bundle))
    previous.load_runtime(previous.OUTPUT / 'runtime/after')
    execution = previous.load_execution(previous.OUTPUT / 'execution/after')
    from hegel_machine.cognition import world_response_transfer as transfer
    original = transfer.export_response_knowledge
    destination = output / case / 'frozen_numeric'

    def capture(state, estimators, source_id):
        # Inputs are final observed responses, not fixture answers. The wrapped
        # export is unchanged and cannot influence the completed trajectory.
        write(destination / 'final_export_inputs.json', {
            'state': deepcopy(state), 'estimators': deepcopy(estimators),
            'source_id': source_id})
        return original(state, estimators, source_id)

    transfer.export_response_knowledge = capture
    execution.run_case(output, case, 'frozen_numeric')
    old = source / case / 'frozen_numeric'
    before, after = read(old / 'brief.json'), read(destination / 'brief.json')
    checks = {
        'actions_equal': read(old / 'action_trace.json') == read(destination / 'action_trace.json'),
        'score_equal': read(old / 'research_trace.json')['score'] == read(destination / 'research_trace.json')['score'],
        'research_costs_equal': all(before['costs'][key] == after['costs'][key] for key in
            ('compute_cost', 'research_compute_units', 'observation_cost', 'environment_steps',
             'model_calls', 'reported_tokens')),
        'export_models_equal': read(old / 'learned_response_knowledge.json')['models'] == read(destination / 'learned_response_knowledge.json')['models'],
        'export_response_count_equal': read(old / 'learned_response_knowledge.json')['source_response_count'] == len(read(destination / 'final_export_inputs.json')['state']['response_settings']),
    }
    write(destination / 'replay_comparison.json', checks)
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=int, choices=range(len(CASES)))
    args = parser.parse_args()
    replay(*CASES[args.index])
