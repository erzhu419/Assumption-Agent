"""Replay recorded choices to isolate when the existing author receives evidence."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]


class DecisionBoundary(BaseException):
    pass


def run(variant, case, directory, *, live=False):
    # The common runtime includes the historical public-view repair. The only
    # experimental difference is the saved automatic-return implementation.
    name = 'hegel_machine.cognition.world_llm_inquiry'
    spec = importlib.util.spec_from_file_location(name,
        OUT / ('runtime_' + variant) / 'world_llm_inquiry.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from scripts import run_response_matched_adoption as matched
    from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
    from hegel_machine.cognition.world_method_transfer import install_method_bundle

    protocol = json.loads((OUT / 'protocol.json').read_text())
    selected = protocol['cases'][case]
    source = ROOT / selected['source']
    original = ROOT / protocol['original_cohort']
    baseline = json.loads((original / 'protocol.json').read_text())
    bundle = json.loads((original / 'frozen_bundle.json').read_text())
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    workspace = directory / 'agent'

    class RecordedThenModel:
        def __init__(self):
            self.count = 0
            self.boundary = None
            self.provider = None

        def next_action(self, request):
            if self.count < selected['recorded_decisions']:
                self.count += 1
                raw = json.loads((source / 'agent' / f'code_action_{self.count:03d}.json').read_text())
                try:
                    args = json.loads(raw['arguments_json'], strict=False)
                except (ValueError, KeyError, TypeError) as error:
                    # Preserve the original provider's recoverable decode result;
                    # do not repair a recorded model answer during replay.
                    return {'action': 'provider_error', 'arguments': {},
                        'error': {'decode_error': f'{type(error).__name__}: {error}'},
                        'usage': {'input_tokens': 0, 'output_tokens': 0,
                                  'cached_input_tokens': 0, 'total_tokens': 0}}
                if raw.get('research_question') is not None:
                    args['research_question'] = raw['research_question']
                if 'research_notes' in request:
                    args['notes'] = raw['research_notes']
                return {'action': raw['action'], 'arguments': args,
                    'usage': {'input_tokens': 0, 'output_tokens': 0,
                              'cached_input_tokens': 0, 'total_tokens': 0}}
            if self.boundary is None:
                self.boundary = {'observations': len(agent.observations),
                    'research_root': request.get('research_root'),
                    'inquiry': request.get('llm_inquiry'),
                    'focus': request.get('research_focus'),
                    'costs': deepcopy(agent.costs)}
                (directory / 'next_decision.json').write_text(json.dumps(self.boundary, indent=2))
                if live:
                    self.provider = matched.CodexCodeProvider(workspace,
                        model=matched.MODEL['requested_alias'],
                        reasoning_effort=matched.MODEL['reasoning_effort'],
                        timeout_seconds=baseline['limits']['provider_request_timeout_seconds'])
            if not live:
                raise DecisionBoundary()
            self.provider.timeout_seconds = min(
                baseline['limits']['provider_request_timeout_seconds'],
                baseline['limits']['task_wall_seconds'] - (monotonic() - agent.started))
            return self.provider.next_action(request)

    provider = RecordedThenModel()
    matched.study.CASES = {'changed_mechanism': baseline['cases']['changed_mechanism']}
    lab = matched.study.TransferSignalLab('changed_mechanism')
    agent = DiscoveryWorldAgent(provider, workspace, arm=selected['arm'], limits=baseline['limits'])
    agent.lab = lab
    install_method_bundle(agent, bundle, lab)

    def checkpoint(data):
        (directory / 'active.json').write_text(json.dumps({
            'status': data['status'], 'costs': data['costs'], 'score': data['score'],
            'recorded_decisions': provider.count, 'live': live}))

    try:
        result = agent.run(lab, on_checkpoint=checkpoint)
        status = result['status']
    except DecisionBoundary:
        # No request was sent for this last counted decision. It is a deliberate
        # observation point, not a provider failure or a new scientific result.
        status = 'next_model_decision'
        result = agent.result()
    result.update(status=status, case='changed_mechanism', repetition=1, arm=selected['arm'])
    result['observations'] = agent.observations
    state = json.loads((workspace / matched.recipe.STATE).read_text())
    result['reference_state'] = {key: state[key] for key in ('test', 'tests') if key in state}
    summary = matched.brief(result, bundle)
    # Recorded choices consume the original decision allowance, but are not API
    # calls and must not enter the cache-hit denominator.
    new_usages = [event['usage'] for event in result.get('events', [])
                  if 'usage' in event][provider.count:]
    reported = [usage['cached_input_tokens'] for usage in new_usages
                if isinstance(usage.get('cached_input_tokens'), int)]
    new_calls = agent.costs['model_calls'] - provider.count if live else 0
    summary['cache'] = {'reported_calls': len(reported),
        'missing_calls': new_calls - len(reported),
        'hit_calls': sum(value > 0 for value in reported),
        'zero_hit_calls': sum(value == 0 for value in reported)}
    original_records = json.loads((source / 'agent/laboratory_observations.json').read_text())
    same_prefix = agent.observations == original_records[:len(agent.observations)]
    summary.update(variant=variant, diagnostic_case=case, recorded_decisions=provider.count,
        live=live, new_model_calls=new_calls,
        next_decision=provider.boundary, original_observation_prefix_equal=same_prefix,
        inquiry_runtime=str(spec.origin))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    (directory / 'active.json').write_text(json.dumps({'status': status, 'finished': True,
        'costs': summary['costs'], 'success': summary['success']}))
    print(json.dumps({k: summary[k] for k in ['variant','diagnostic_case','status','success',
        'recorded_decisions','new_model_calls','original_observation_prefix_equal','costs']}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['before','after'])
    parser.add_argument('case')
    parser.add_argument('directory', type=Path)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    run(args.variant, args.case, args.directory, live=args.live)
