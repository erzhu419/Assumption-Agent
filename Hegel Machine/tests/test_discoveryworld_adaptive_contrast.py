"""Adaptive contrasts use real step feedback; fixtures do not establish LLM adoption."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_contrast import (
    PATH as FIXED_PATH,
    SOURCE as FIXED_SOURCE,
    contrast as fixed_contrast,
    prepared as fixed_prepared,
)


PATH = "shared/reference_contrast.py"
QUESTION = {"id": "physical", "question": "Does restoring the control restore the response?",
            "parent_id": "task", "depends_on": []}
SOURCE = '''def design(spec, observations, memory):
    return {'hypotheses': [
        {'id': 'control', 'claim': 'The observed control alone restores the response'},
        {'id': 'extra', 'claim': 'Restoration requires an additional operation'}],
        'experiments': [{'id': 'restore',
            'outcome': 'A successful baseline changes under perturbation and returns on restoration',
            'procedure': {'path': memory['program_path'], 'max_steps': 8},
            'predictions': {'control': True, 'extra': False}}]}

def trial(memory, observations):
    question = next(q for q in memory['questions'] if q['id'] == 'physical')
    choice = question['experiment_choice']
    assert choice['selected_experiment']['predictions'] == {'control': True, 'extra': False}
    return [r for r in observations if r['index'] >= choice['started_before_observation']]

def select_actions(spec, observations, memory):
    rows = trial(memory, observations)
    if len(rows) >= 3:
        return {'stop': 'The baseline, perturbation and restoration were observed'}
    if not rows:
        return [{'action': 'TALK', 'arg1': 7}]
    current = spec['current_observation']
    assert current['setting'] == rows[-1]['observation']['setting']
    wanted = 'Increase by 1' if len(rows) == 1 else 'Decrease by 1'
    option = next(k for k, v in current['dialog_box']['dialogOptions'].items() if v == wanted)
    return [{'chosen_dialog_option_int': int(option)}]

def answer(spec, observations, memory):
    rows = trial(memory, observations)
    complete = (len(rows) == 3
        and all(r['feedback']['success'] for r in rows)
        and rows[0]['observation']['setting'] == rows[2]['observation']['setting']
        and rows[1]['observation']['setting'] != rows[0]['observation']['setting']
        and [r['observation']['active'] for r in rows] == [True, False, True])
    return {'status': 'supported' if complete else 'inconclusive',
            'answer': {'restored': complete}, 'evidence': [r['index'] for r in rows]}
'''


class MenuLab:
    """The control must be read from a changing public menu after each action."""

    def __init__(self, *, fail_adjustment=False, baseline_active=True):
        self.environment_steps = 0
        self.actions = []
        self.setting = 5.5
        self.fail_adjustment = fail_adjustment
        self.baseline_active = baseline_active
        self.frozen_choices = []

    def public_observation(self):
        options = ({"4": "Increase by 1", "8": "Decrease by 1"}
                   if self.setting == 5.5 else {"2": "Decrease by 1", "6": "Increase by 1"})
        return {"taskProgress": [{"description": "Explain and control the response", "completed": False}],
                "setting": self.setting,
                "active": self.baseline_active and self.setting == 5.5,
                "dialog_box": {"is_in_dialog": bool(self.actions), "dialogOptions": options}}

    def capabilities(self):
        return {"actions": ["TALK"]}

    def act(self, action):
        question = next(q for q in self.agent.questions if q["id"] == "physical")
        self.frozen_choices.append(deepcopy(question["experiment_choice"]))
        success = True
        if "chosen_dialog_option_int" in action:
            option = str(action["chosen_dialog_option_int"])
            label = self.public_observation()["dialog_box"]["dialogOptions"].get(option)
            success = not self.fail_adjustment and label is not None
            if success:
                self.setting += 1 if label.startswith("Increase") else -1
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        return {"action": deepcopy(action), "feedback": {"success": success},
                "observation": self.public_observation()}

    def score(self):
        return {"completed": False, "completedSuccessfully": False, "scoreNormalized": 0.}


def prepared(tmp_path, *, arm="hegel", **lab_options):
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm, limits=LIMITS)
    lab = MenuLab(**lab_options)
    agent.lab, lab.agent = lab, agent
    agent._question({"id": "task", "question": "Explain and control the response"})
    return agent, lab


def contrast(agent, lab, source=SOURCE):
    return agent._execute("run", {"operator": "contrast", "path": PATH,
                                  "code": source, "question": QUESTION}, lab)


def question(agent, qid="physical"):
    return next(q for q in agent.questions if q["id"] == qid)


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_selected_procedure_reads_live_control_and_restores_physical_response(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm=arm)
    result = contrast(agent, lab)
    assert result["selected_experiment"]["id"] == "restore"
    assert lab.actions == [{"action": "TALK", "arg1": 7},
                           {"chosen_dialog_option_int": 4}, {"chosen_dialog_option_int": 2}]
    assert question(agent)["status"] == "supported"
    assert question(agent)["answer"] == {"restored": True}
    assert question(agent)["evidence"] == question(agent)["acquired_observations"] == [0, 1, 2]
    assert question(agent)["experiment_path"] == PATH
    assert all(choice == lab.frozen_choices[0] for choice in lab.frozen_choices)
    assert lab.frozen_choices[0]["started_before_observation"] == 0
    assert len(agent.method_results) == 4  # Three feedback-driven actions and explicit stop.
    assert agent.child_answers_returned == 1
    assert question(agent, "task")["status"] == "reopened"


def test_a_repeated_design_does_not_mistake_inherited_trial_for_new_execution(tmp_path):
    agent, lab = prepared(tmp_path)
    contrast(agent, lab)
    contrast(agent, lab)
    assert len(lab.actions) == 6
    assert question(agent)["evidence"] == [3, 4, 5]
    assert question(agent)["experiment_choice"]["started_before_observation"] == 3


@pytest.mark.parametrize("case", ["early_stop", "baseline_unmet"])
def test_explicit_stop_does_not_supply_scientific_support(tmp_path, case):
    agent, lab = prepared(tmp_path, baseline_active=case != "baseline_unmet")
    source = SOURCE.replace("len(rows) >= 3", "len(rows) >= 1") if case == "early_stop" else SOURCE
    contrast(agent, lab, source)
    assert question(agent)["status"] == "inconclusive"
    assert question(agent)["answer"] == {"restored": False}
    assert question(agent)["evidence"] == list(range(len(lab.actions)))


@pytest.mark.parametrize("case", ["failed_action", "step_limit", "environment_limit"])
def test_interrupted_procedure_keeps_paid_rows_without_an_automatic_answer(tmp_path, case):
    agent, lab = prepared(tmp_path, fail_adjustment=case == "failed_action")
    source = SOURCE.replace("'max_steps': 8", "'max_steps': 2") if case == "step_limit" else SOURCE
    if case == "environment_limit":
        agent.limits["environment_steps"] = 2
    contrast(agent, lab, source)
    assert len(lab.actions) == 2
    assert question(agent)["acquired_observations"] == [0, 1]
    assert question(agent)["experiment_path"] == PATH
    assert "answer" not in question(agent)
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "analysis" and focus["evidence"] == [0, 1]
    assert focus["path"] == "shared/analysis_physical.py"


def test_answer_repair_preserves_executable_design_after_failed_acquisition(tmp_path):
    agent, lab = prepared(tmp_path, fail_adjustment=True)
    contrast(agent, lab)
    focus = agent._request()["research_focus"]
    repair = """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'The control failed; no restoration was tested',
            'evidence': [observations[-1]['index']]}
"""
    agent._execute("write", {"path": focus["path"], "code": repair}, lab)
    assert (tmp_path / PATH).read_text() == SOURCE
    assert agent._program(PATH, "design")["experiments"][0]["id"] == "restore"
    assert question(agent)["status"] == "inconclusive"
    assert question(agent)["analysis_path"] == "shared/analysis_physical.py"
    lab.fail_adjustment = False
    agent._execute("run", {"operator": "contrast", "path": PATH, "question": QUESTION}, lab)
    assert len(lab.actions) == 5
    assert question(agent)["evidence"] == [4]
    assert question(agent)["status"] == "inconclusive"  # The retained repair, not the old answer, ran.
    assert question(agent)["analysis_path"] == "shared/analysis_physical.py"


def test_fixed_action_contrast_also_retains_design_during_answer_repair(tmp_path):
    agent, lab = fixed_prepared(tmp_path, failed=True)
    fixed_contrast(agent, lab)
    assert question(agent)["experiment_path"] == FIXED_PATH
    focus = agent._request()["research_focus"]
    assert focus["path"] != FIXED_PATH
    agent._execute("write", {"path": focus["path"], "code": """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'Physical test was not executed', 'evidence': [1]}
"""}, lab)
    assert (tmp_path / FIXED_PATH).read_text() == FIXED_SOURCE
    assert len(agent._program(FIXED_PATH, "design")["experiments"]) == 2


def test_separate_design_write_replaces_old_repaired_analysis(tmp_path):
    agent, lab = prepared(tmp_path, fail_adjustment=True)
    contrast(agent, lab)
    focus = agent._request()["research_focus"]
    agent._execute("write", {"path": focus["path"], "code": """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'Old failed control analysis',
            'evidence': [observations[-1]['index']]}
"""}, lab)
    assert question(agent)["analysis_path"] == "shared/analysis_physical.py"

    revised = SOURCE.replace(
        "The observed control alone restores the response",
        "The observed control restores the response under the newly tested conditions",
    ).replace("'answer': {'restored': complete}", "'answer': {'revised_restored': complete}")
    agent._execute("write", {"path": PATH, "code": revised}, lab)
    lab.fail_adjustment = False
    agent._execute("run", {"operator": "contrast", "path": PATH, "question": QUESTION}, lab)

    assert question(agent)["analysis_path"] == PATH
    assert question(agent)["status"] == "supported"
    assert question(agent)["answer"] == {"revised_restored": True}
    assert question(agent)["evidence"] == [2, 3, 4]
    assert (tmp_path / "shared/analysis_physical.py").is_file()


def test_authored_subquestion_takes_precedence_over_contrast_fallback(tmp_path):
    agent, lab = prepared(tmp_path)
    source = SOURCE.replace("'max_steps': 8", "'max_steps': 1").replace(
        "return [{'action': 'TALK', 'arg1': 7}]",
        "return {'question': {'id': 'readout', 'question': 'Read the baseline', "
        "'parent_id': 'physical'}, 'actions': [{'action': 'TALK', 'arg1': 7}]}")
    contrast(agent, lab, source)
    assert question(agent, "readout")["acquired_observations"] == [0]
    assert question(agent)["acquired_observations"] == []
    assert question(agent, "readout")["parent_id"] == "physical"
    assert "answer" not in question(agent)
