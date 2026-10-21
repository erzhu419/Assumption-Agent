"""Preparing a shared registered input twice preserves its frozen source."""

from copy import deepcopy
import json

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts import run_anonymous_revision_inquiry as study


def test_same_output_preparation_reuses_the_original_frozen_library(tmp_path):
    original = tmp_path / 'source_initial_library.json'
    original.write_text('{"candidates":{}}')
    prior = tmp_path / 'frozen_pr3_library.json'
    prior.write_text('{"candidates":{}}')
    knowledge = {'files': {'shared/old_predictor.py': 'def predict(row): return 0.\n'},
                 'lessons': ['Retain the original failed source.'],
                 'theory_consolidation': {'file': str(original), 'json_path': []}}
    unchanged = deepcopy(knowledge)
    config = InquiryLLMConfig(max_model_calls=0, token_budget=0)
    plan = study.protocol(kinds=('local_fixture',), arms=study.ORGANIZATION_ARMS,
        config=config, research_organization={'phase': 'target', 'theories': [],
                                             'pr3_library_path': str(prior)})
    output = tmp_path / 'target'
    output.mkdir()
    first_registration, prepared = study._prepare_organization(plan, output, knowledge)
    fixed = output / 'initial_theory_library.json'
    first_bytes = fixed.read_bytes()
    pr3_bytes = (output / 'pr3_theory_library.json').read_bytes()
    assert json.loads(first_bytes) == {'source_ref': unchanged['theory_consolidation']}

    # The real launcher prepares the shared remove input, then passes it to
    # run_study, which prepares that same output again before constructing ABC.
    second_registration, repeated = study._prepare_organization(plan, output, prepared)

    assert fixed.read_bytes() == first_bytes
    assert (output / 'pr3_theory_library.json').read_bytes() == pr3_bytes
    assert second_registration == first_registration
    assert repeated == prepared
    assert knowledge == unchanged
    assert original.read_text() == prior.read_text() == '{"candidates":{}}'
    for group in study.ORGANIZATION_ARMS:
        agent = study.RegisteredOrganizationAgent(None, output / group, arm=group,
            knowledge=deepcopy(repeated), research_organization=second_registration,
            config=config)
        assert agent.theory_consolidation['candidates'] == {}
        assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
        assert agent.knowledge['files']['shared/old_predictor.py']
