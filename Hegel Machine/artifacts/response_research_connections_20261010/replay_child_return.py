"""Replay twelve recorded decisions through the repaired production loop.

No forced handoff, new model decision, or scientific answer is supplied. The
handoff_only variant uses the saved pre-integration runtime with just the changed
inquiry loop; integrated overlays all repaired modules saved in runtime_after.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]


class DecisionBoundary(BaseException):
    pass


def main(variant):
    import hegel_machine.cognition as cognition
    cognition.__path__.insert(0, str(HERE / 'runtime_before'))
    if variant == 'handoff_only':
        name = 'hegel_machine.cognition.world_llm_inquiry'
        source = HERE / 'runtime_after/world_llm_inquiry.py'
        if not source.exists():
            source = ROOT / 'src/hegel_machine/cognition/world_llm_inquiry.py'
        spec = importlib.util.spec_from_file_location(
            name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        cognition.__path__.insert(0, str(HERE / 'runtime_after'))

    from scripts import run_response_matched_adoption as matched
    from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
    from hegel_machine.cognition.world_method_transfer import install_method_bundle

    source = ROOT / 'artifacts/response_refuted_author_return_20261010/live/repeat_02/after/agent'
    original = ROOT / 'artifacts/response_current_public_view_live_20261010'
    prefix = original / 'changed_mechanism/repeat_03/hegel/agent'
    protocol = json.loads((original / 'protocol.json').read_text())
    bundle = json.loads((original / 'frozen_bundle.json').read_text())
    output = HERE / variant
    output.mkdir(parents=True, exist_ok=False)

    class Recorded:
        count = 0

        def next_action(self, request):
            if self.count == 12:
                raise DecisionBoundary()
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

    provider = Recorded()
    matched.study.CASES = {'changed_mechanism': protocol['cases']['changed_mechanism']}
    lab = matched.study.TransferSignalLab('changed_mechanism')
    agent = DiscoveryWorldAgent(provider, output / 'agent', arm='hegel', limits=protocol['limits'])
    agent.lab = lab
    install_method_bundle(agent, bundle, lab)
    try:
        result = agent.run(lab)
        status = result['status']
    except DecisionBoundary:
        result = agent.result()
        status = 'next_unprovided_model_decision'
    result.update(status=status, case='changed_mechanism', repetition=2, arm='hegel',
                  observations=agent.observations)
    state = json.loads((agent.workspace / matched.recipe.STATE).read_text())
    result['reference_state'] = {key: state[key] for key in ('test', 'tests') if key in state}
    summary = matched.brief(result, bundle)
    old_observations = json.loads((source / 'laboratory_observations.json').read_text())
    same = next((i for i, (a, b) in enumerate(zip(agent.observations, old_observations))
                 if a != b), min(len(agent.observations), len(old_observations)))
    summary.update(variant=variant, recorded_decisions=provider.count, new_model_calls=0,
                   forced_handoff=False, identical_prefix_observations=same,
                   original_child_return_prefix_equal=same >= 237)
    summary['cache'] = dict(reported_calls=0, missing_calls=0, hit_calls=0, zero_hit_calls=0)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ('variant', 'status', 'success',
        'recorded_decisions', 'new_model_calls', 'identical_prefix_observations', 'costs')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=('handoff_only', 'integrated'))
    main(parser.parse_args().variant)
