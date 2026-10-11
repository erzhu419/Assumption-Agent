"""Offline intervention at one recorded child return; no production code change.

Reproduce the first twelve original choices. Compare the unchanged next-decision
boundary with an operator-imposed return to the existing parent at that boundary.
No family, setting, model reply or additional decision is supplied after it.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from scripts import run_response_matched_adoption as matched
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_method_transfer import install_method_bundle


class Boundary(BaseException):
    pass


def main(variant):
    source = HERE / 'live/repeat_02/after/agent'
    original = ROOT / 'artifacts/response_current_public_view_live_20261010'
    prefix = original / 'changed_mechanism/repeat_03/hegel/agent'
    protocol = json.loads((original / 'protocol.json').read_text())
    bundle = json.loads((original / 'frozen_bundle.json').read_text())
    output = HERE / 'child_return_audit' / variant
    output.mkdir(parents=True, exist_ok=False)
    snapshots = []

    class Recorded:
        count = 0

        def next_action(self, request):
            if self.count == 12:
                raise Boundary()
            self.count += 1
            folder = prefix if self.count <= 3 else source
            number = self.count if self.count <= 3 else self.count - 3
            raw = json.loads((folder / f'code_action_{number:03d}.json').read_text())
            usage = dict(input_tokens=0, output_tokens=0, cached_input_tokens=0, total_tokens=0)
            try:
                arguments = json.loads(raw['arguments_json'], strict=False)
            except (ValueError, KeyError, TypeError) as error:
                return dict(action='provider_error', arguments={},
                            error={'decode_error': str(error)}, usage=usage)
            if raw.get('research_question') is not None:
                arguments['research_question'] = raw['research_question']
            if 'research_notes' in request:
                arguments['notes'] = raw['research_notes']
            return dict(action=raw['action'], arguments=arguments, usage=usage)

    class AuditedAgent(DiscoveryWorldAgent):
        def _procedure_terminal_status(self):
            if (not snapshots and self.costs['model_calls'] == 12 and self._inquiry_stack
                    and self._inquiry_stack[-1]['question_id'] == 'reference_extension_5'
                    and self.llm_inquiries.get('reference_extension_5_signal_cubic', {}).get('status') == 'answered'):
                frame = self._inquiry_stack[-1]
                state = json.loads((self.workspace / matched.recipe.STATE).read_text())
                snapshot = {'observations': len(self.observations), 'costs': deepcopy(self.costs),
                    'pending_dependencies': self._pending_dependencies(frame['question']),
                    'parent_program_path': frame['assignment'].get('parent_program_path'),
                    'response_research': state.get('response_research'),
                    'child': deepcopy(next(q for q in self.questions
                        if q['id'] == 'reference_extension_5_signal_cubic')),
                    'original_observation_prefix_equal': self.observations ==
                        json.loads((source / 'laboratory_observations.json').read_text())[:len(self.observations)]}
                snapshots.append(snapshot)
                (output / 'child_return_state.json').write_text(json.dumps(snapshot, indent=2))
                # This is the sole diagnostic intervention. It changes control
                # ownership at this observed boundary, not a scientific answer.
                if variant == 'yield_parent':
                    frame['stop_requested'] = True
            return super()._procedure_terminal_status()

    provider = Recorded()
    matched.study.CASES = {'changed_mechanism': protocol['cases']['changed_mechanism']}
    lab = matched.study.TransferSignalLab('changed_mechanism')
    agent = AuditedAgent(provider, output / 'agent', arm='hegel', limits=protocol['limits'])
    agent.lab = lab
    install_method_bundle(agent, bundle, lab)
    try:
        result = agent.run(lab)
        status = result['status']
    except Boundary:
        result = agent.result()
        status = 'next_unprovided_model_decision'
    result.update(status=status, case='changed_mechanism', repetition=2, arm='hegel',
                  observations=agent.observations)
    state = json.loads((agent.workspace / matched.recipe.STATE).read_text())
    result['reference_state'] = {key: state[key] for key in ('test', 'tests') if key in state}
    summary = matched.brief(result, bundle)
    summary.update(variant=variant, recorded_decisions=provider.count, new_model_calls=0,
                   intervention_applied=bool(snapshots) and variant == 'yield_parent',
                   child_return_observations=snapshots[0]['observations'] if snapshots else None,
                   prefix_equal=snapshots[0]['original_observation_prefix_equal'] if snapshots else None)
    summary['cache'] = dict(reported_calls=0, missing_calls=0, hit_calls=0, zero_hit_calls=0)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ('variant', 'status', 'success',
        'recorded_decisions', 'new_model_calls', 'child_return_observations', 'prefix_equal', 'costs')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=('unchanged', 'yield_parent'))
    main(parser.parse_args().variant)
