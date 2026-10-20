"""Archive/runtime regressions; scripted actions are not discovery evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry import InquiryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_paired_method import ProcedureConsumerAgent


MODEL = "def predict(row):\n return float(next(iter(row['features'].values())))\n"
PLAN = """def select_actions(spec, observations, memory):
    controls = spec['controls']
    return {'actions': [{'kind': 'response', 'point': dict(zip(controls, values))}
                        for values in [(0.2, -0.1), (-0.4, 0.5)]]}
"""
MANIFEST = {
    "hypotheses": [{"id": "response", "path": "candidate.py"}],
    "research_programs": [{"id": "probe", "path": "shared/probe.py"}],
    "support_questions": [{"id": "q1", "evidence": [0, 1]}],
}
QUESTION = {"id": "q1", "question": "Does the retained prediction still hold?",
            "parent_id": None, "depends_on": []}


def decision(action, arguments):
    return {"action": action, "arguments": arguments,
            "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}


class ArchiveProvider:
    """Use real write/run/query/finish paths without making external calls."""

    def __init__(self):
        self.memories = []

    def next_action(self, request):
        self.memories.append(deepcopy(request["session_knowledge"]))
        if not request["events"] and not request["session_knowledge"]:
            return decision("run", {"code": "\n".join([
                "from pathlib import Path",
                "Path('shared').mkdir(exist_ok=True)",
                f"Path('candidate.py').write_text({MODEL!r})",
                f"Path('shared/probe.py').write_text({PLAN!r})",
                f"Path('shared/knowledge.json').write_text({json.dumps(MANIFEST)!r})",
                "Path('shared/notes.md').write_text('Candidate and experiment archive.')",
            ])})
        if not request["events"] or request["events"][-1]["action"] == "run":
            return decision("query", {"path": "shared/probe.py", "question": QUESTION})
        return decision("finish", {"model": "candidate.py", "lesson": "A paid check supported this candidate."})


def laboratory():
    return InquiryLab(seed=8, mechanism="linear", driver=0, orientation=1)


def configuration(**updates):
    return InquiryLLMConfig(**{"observation_budget": 10, "confirmation_rows": 3,
                              "max_model_calls": 4, **updates})


def test_code_exports_executed_archive_and_both_consumers_receive_it_unchanged(tmp_path):
    source = CommonArchiveInquiryAgent(ArchiveProvider(), tmp_path / "source",
        arm="code_react", config=configuration())
    result = source.run(laboratory())
    assert result["status"] == "completed"
    assert result["fresh_confirmation"]["passed"]
    memory = result["outgoing_knowledge"]
    assert memory["files"]["candidate.py"] == MODEL
    assert memory["files"]["shared/probe.py"] == PLAN
    assert memory["manifest"] == MANIFEST
    assert memory["lessons"] == ["A paid check supported this candidate."]
    assert memory["question_traces"][0]["id"] == "q1"
    before = deepcopy(memory)

    for arm in ("code_react", "hegel"):
        provider = ArchiveProvider()
        consumer = CommonArchiveInquiryAgent(provider, tmp_path / arm, arm=arm,
            config=configuration(), knowledge=memory, source_task=False)
        consumed = consumer.run(laboratory())
        assert provider.memories[0] == before
        assert consumer.knowledge == before
        assert consumed["status"] == "completed"
        assert consumed["events"][0]["feedback"]["inherited_program"]
        assert consumed["fresh_confirmation"]["passed"]
        assert consumed["outgoing_knowledge"]["files"]["candidate.py"] == MODEL
        assert consumed["outgoing_knowledge"]["question_traces"][:1] == memory["question_traces"]
    assert memory == before


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_unfinished_source_keeps_generated_candidate_manifest_and_references(tmp_path, arm):
    class UnfinishedProvider:
        def next_action(self, request):
            manifest = deepcopy(MANIFEST)
            manifest["hypotheses"][0]["parameters"] = {"path": "models/parameters.json"}
            return decision("run", {"code": "\n".join([
                "from pathlib import Path",
                "Path('shared').mkdir(exist_ok=True)",
                "Path('models').mkdir(exist_ok=True)",
                f"Path('candidate.py').write_text({MODEL!r})",
                f"Path('shared/probe.py').write_text({PLAN!r})",
                "Path('models/parameters.json').write_text('{\"coefficient\": 1.0}')",
                f"Path('shared/knowledge.json').write_text({json.dumps(manifest)!r})",
                "raise ValueError('unfinished source investigation')",
            ])})

    agent = CommonArchiveInquiryAgent(UnfinishedProvider(), tmp_path, arm=arm,
                                     config=configuration(max_model_calls=1))
    result = agent.run(laboratory())
    assert result["status"] == "model_call_budget_exhausted"
    assert result["model"] is None and result["fresh_confirmation"] is None
    assert result["events"][0]["feedback"]["returncode"] != 0
    outgoing = result["outgoing_knowledge"]
    assert outgoing["files"]["candidate.py"] == MODEL
    assert outgoing["files"]["shared/probe.py"] == PLAN
    assert json.loads(outgoing["files"]["models/parameters.json"]) == {"coefficient": 1.0}
    assert outgoing["manifest"]["hypotheses"][0]["path"] == "candidate.py"
    assert outgoing["lessons"] == []


def test_partial_manifest_does_not_drop_failed_source_checkpoint(tmp_path):
    class PartialProvider:
        def next_action(self, request):
            return decision("write", {"path": "shared/knowledge.json", "code": '{"hypotheses":'})

    agent = CommonArchiveInquiryAgent(PartialProvider(), tmp_path, arm="hegel",
                                     config=configuration(max_model_calls=1))
    result = agent.run(laboratory())
    assert result["outgoing_knowledge"]["files"]["shared/knowledge.json"] == '{"hypotheses":'
    assert result["outgoing_knowledge"]["manifest"] == {}


def test_retained_predictor_transfers_its_fitting_rows_and_parameter_file(tmp_path):
    source = CommonArchiveInquiryAgent(None, tmp_path / "source", arm="code_react",
                                      config=configuration())
    lab = laboratory()
    source._execute("write", {"path": "shared/probe.py", "code": PLAN}, lab)
    source._execute("query", {"path": "shared/probe.py", "question": QUESTION}, lab)
    source._execute("write", {"path": "parameters.json", "code": '{"offset": 3}'}, lab)
    fitted = """import json
def fit(rows):
    global coefficient, offset
    offset = json.load(open('parameters.json'))['offset']
    coefficient = sum(row['target'] / next(iter(row['features'].values()))
                      for row in rows) / len(rows) + offset
def predict(row):
    return (coefficient - offset) * next(iter(row['features'].values()))
"""
    source._execute("write", {"path": "model.py", "code": fitted}, lab)
    manifest = deepcopy(MANIFEST)
    manifest["hypotheses"][0]["path"] = "model.py"
    source._execute("write", {"path": "shared/knowledge.json", "code": json.dumps(manifest)}, lab)
    host_records = ("active.json", "training.json", "archive.json", "evaluation.json",
                    "result.json", "protocol.json", "summary.json",
                    "interrupted_training.json", "interrupted_training_0877.json")
    checkpoint = json.dumps(source.result("researching"))
    for name in host_records:
        (source.workspace / name).write_text(checkpoint)
    evaluated = source._execute("evaluate", {"model": "model.py"}, lab)
    assert evaluated["mse"] == pytest.approx(0.)
    retained = evaluated["evaluated_model"]["model"]
    (source.workspace / "code_action_001.json").write_text('{"provider": "private trace"}')
    memory = source.result("unfinished")["outgoing_knowledge"]
    assert "_evaluated_model/model_observations.json" in memory["files"]
    assert "_evaluated_model/parameters.json" in memory["files"]
    assert "parameters.json" in memory["files"]
    assert "code_action_001.json" not in memory["files"]
    assert "research_state.json" not in memory["files"]
    for name in host_records:
        assert name not in memory["files"]
        assert f"_evaluated_model/{name}" not in memory["files"]
        assert not (source.workspace / "_evaluated_model" / name).exists()
        assert (source.workspace / name).read_text() == checkpoint

    for arm in ("code_react", "hegel"):
        consumer = CommonArchiveInquiryAgent(None, tmp_path / arm, arm=arm,
            knowledge=memory, config=configuration(), source_task=False)
        # Current-task fitting rows start empty. Only the saved source view can
        # support this predictor's fit(), and its parameters must also transfer.
        assert json.loads((consumer.workspace / "model_observations.json").read_text()) == []
        feedback = consumer._execute("finish", {"model": retained}, laboratory())
        assert feedback["fresh_confirmation"]["passed"]
        assert feedback["fresh_confirmation"]["mse"] == pytest.approx(0.)
        assert consumer.costs["observation_cost"] == 3


def test_live_checkpoints_do_not_recursively_enter_the_common_archive(tmp_path):
    checkpoints = []

    def save_checkpoint(result):
        checkpoints.append(deepcopy(result))
        (tmp_path / "active.json").write_text(json.dumps(result))

    agent = CommonArchiveInquiryAgent(ArchiveProvider(), tmp_path, arm="code_react",
                                     config=configuration())
    result = agent.run(laboratory(), on_checkpoint=save_checkpoint)
    assert result["status"] == "completed"
    assert len(checkpoints) == 3
    assert (tmp_path / "active.json").is_file()
    for checkpoint in [*checkpoints, result]:
        assert "active.json" not in checkpoint["outgoing_knowledge"]["files"]
        assert "candidate.py" in checkpoint["outgoing_knowledge"]["files"]
        assert "model_observations.json" in checkpoint["outgoing_knowledge"]["files"]


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_common_archive_does_not_erase_hegel_manifest_requirement(tmp_path, arm):
    class PlainProvider:
        def next_action(self, request):
            if not request["events"]:
                return decision("write", {"path": "model.py", "code": MODEL})
            return decision("finish", {"model": "model.py"})

    agent = CommonArchiveInquiryAgent(PlainProvider(), tmp_path, arm=arm,
                                     config=configuration(max_model_calls=2))
    result = agent.run(laboratory())
    if arm == "code_react":
        assert result["status"] == "completed"
        assert result["fresh_confirmation"]["passed"]
    else:
        assert result["model"] is None
        assert "Hegel memory requires" in result["events"][-1]["feedback"]["error"]


def test_original_code_condition_and_new_public_contract_remain_distinct(tmp_path):
    old = ProcedureConsumerAgent(None, tmp_path / "old", arm="code_react")
    old._execute("write", {"path": "shared/model.py", "code": MODEL}, laboratory())
    assert old.result("unfinished")["outgoing_knowledge"] == {}
    assert "Code/ReAct receives no previous-task knowledge" in old.common_interface
    common = CommonArchiveInquiryAgent.common_interface
    assert "Both Code/ReAct and Hegel receive the same supplied archive" in common
    assert "Code/ReAct receives no previous-task knowledge" not in common
    assert "operator:'procedure'" in common
    assert "def select_actions(spec, observations, memory)" not in common
    assert "hysteresis" not in common.lower()


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_string_question_returns_feedback_then_accepts_authored_correction(tmp_path, arm):
    # Real cohort replies Code005/004 and Code009/019 used this string where
    # the existing interface requires a question object. Neither action ran.
    identifier = "multivariate_latent_quadratic_update"
    requests = []

    class CorrectingProvider:
        def next_action(self, request):
            requests.append(deepcopy(request))
            (tmp_path / "source" / "research_context.json").write_text(json.dumps(request))
            actions = [
                decision("evaluate", {"model": "model.py", "code": MODEL,
                                      "research_question": identifier}),
                decision("write", {"path": "model.py", "code": MODEL,
                    "research_question": {**QUESTION, "id": identifier}}),
                decision("finish", {"model": "model.py"}),
            ]
            return actions[len(requests) - 1]

    agent = CommonArchiveInquiryAgent(CorrectingProvider(), tmp_path / "source",
        arm=arm, source_task=False, config=configuration(max_model_calls=3))
    assert len(agent.questions) == (1 if arm == "hegel" else 0)
    result = agent.run(laboratory())
    assert result["status"] == "completed"
    assert result["fresh_confirmation"]["passed"]
    error = result["events"][0]["feedback"]["error"]
    assert error.startswith("ValueError:") and "JSON object" in error
    assert requests[1]["events"][-1]["feedback"]["error"] == error
    assert "model.py" not in requests[1]["files"]
    assert len(result["events"]) == result["costs"]["model_calls"] == 3
    assert result["costs"]["reported_tokens"] == 3
    assert result["costs"]["observation_cost"] == 3
    corrected = next(q for q in result["online_questions"] if q["id"] == identifier)
    assert corrected["question"] == QUESTION["question"]

    # The invalid paid reply remains in history and must not break resumption.
    restored = CommonArchiveInquiryAgent(None, tmp_path / "restored", arm=arm,
        source_task=False, config=configuration(max_model_calls=4))
    restored.restore_research(result, agent.workspace, laboratory(),
                              confirmation_seed=9173)
    assert restored.events == result["events"]
    assert restored.costs == result["costs"]
    assert restored.questions == result["online_questions"]
