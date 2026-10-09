"""A saved revised experiment uses its answer without discarding independent repairs."""

from copy import deepcopy

import pytest

import test_discoveryworld_adaptive_contrast as adaptive
import test_discoveryworld_contrast as fixed


NEW_PATH = "shared/revised_contrast.py"
REPAIR_PATH = "shared/independent_answer.py"
REPAIR = """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': {'independent_analysis': True},
            'evidence': [observations[-1]['index']]}
"""


def state(agent):
    return next(q for q in agent.questions if q["id"] == "physical")


def revise_at_new_path(agent, lab, source):
    question = {"id": "physical", "question": "Does the response hold under the revised experiment?"}
    written = agent._execute("write", {"path": NEW_PATH, "code": source,
                                       "research_question": question}, lab)
    return written


@pytest.mark.parametrize("case", [fixed, adaptive], ids=["fixed_actions", "adaptive_procedure"])
@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_new_saved_design_adopts_its_own_answer_in_both_contrast_execution_paths(tmp_path, case, arm):
    agent, lab = case.prepared(tmp_path, arm=arm)
    case.contrast(agent, lab)
    previous = deepcopy(state(agent))
    previous_actions = len(lab.actions)
    source = case.SOURCE.replace("'answer': {", "'answer': {'revision': 'new', ")
    written = revise_at_new_path(agent, lab, source)
    assert written["contrast_action"]["arguments"]["path"] == NEW_PATH

    agent._execute("run", {"operator": "contrast", "path": NEW_PATH,
                           "question": {"id": "physical"}}, lab)

    current = state(agent)
    assert current["design_path"] == current["analysis_path"] == NEW_PATH
    assert current["answer"]["revision"] == "new"
    assert current["status"] == previous["status"]
    assert current["evidence"] == list(range(previous_actions, len(lab.actions)))
    assert len(lab.actions) > previous_actions
    assert (tmp_path / case.PATH).read_text() == case.SOURCE
    assert (tmp_path / NEW_PATH).read_text() == source


@pytest.mark.parametrize("case", [fixed, adaptive], ids=["fixed_actions", "adaptive_procedure"])
def test_new_saved_design_preserves_a_separately_authored_repaired_answer(tmp_path, case):
    agent, lab = case.prepared(tmp_path)
    case.contrast(agent, lab)
    agent._execute("evaluate", {"question_id": "physical", "path": REPAIR_PATH, "code": REPAIR}, lab)
    source = case.SOURCE.replace("'answer': {", "'answer': {'revision': 'new', ")
    revise_at_new_path(agent, lab, source)

    agent._execute("run", {"operator": "contrast", "path": NEW_PATH,
                           "question": {"id": "physical"}}, lab)

    current = state(agent)
    assert current["design_path"] == NEW_PATH and current["analysis_path"] == REPAIR_PATH
    assert current["status"] == "inconclusive" and current["answer"] == {"independent_analysis": True}
    assert current["evidence"] == [len(lab.actions) - 1]
    assert (tmp_path / REPAIR_PATH).read_text() == REPAIR


@pytest.mark.parametrize("case", [fixed, adaptive], ids=["fixed_actions", "adaptive_procedure"])
def test_new_design_without_an_answer_keeps_the_existing_analysis(tmp_path, case):
    agent, lab = case.prepared(tmp_path)
    case.contrast(agent, lab)
    previous = deepcopy(state(agent)["answer"])
    source = case.SOURCE.split("\ndef answer(")[0]
    revise_at_new_path(agent, lab, source)

    agent._execute("run", {"operator": "contrast", "path": NEW_PATH,
                           "question": {"id": "physical"}}, lab)

    current = state(agent)
    assert current["design_path"] == NEW_PATH and current["analysis_path"] == case.PATH
    assert current["answer"] == previous
