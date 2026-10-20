"""Parent task retention across an executed negative child and evidence revision.

The affine model and scripted choices are test fixtures, never live Luna inputs.
"""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.anonymous_method_learning import (
    SourceProcedureRepairLearner, paid_scene_trial_preparer,
)
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_evidence_procedure_execution import PARENT, PROGRAM, procedure_source
from test_executable_evidence_binding import return_local_checks, return_region
from test_inquiry_revision import action
from test_source_research_work import OLD_METHOD
from test_temporal_python_refit import LawLab


def test_negative_child_returns_to_same_assignment_before_evidence_revision(tmp_path):
    knowledge = {"files": {METHOD_PATH: OLD_METHOD, "shared/knowledge.json":
        json.dumps({"hypotheses": [], "research_programs": [], "support_questions": []})}}
    limits = InquiryLLMConfig(observation_budget=60, compute_budget=10000,
        confirmation_rows=3, execution_seconds=60., max_model_calls=0, token_budget=0)
    lab = LawLab()
    source = CommonArchiveInquiryAgent(None, tmp_path / "source", arm="hegel",
        source_task=False, knowledge=deepcopy(knowledge), config=limits)
    source._execute("query", {"actions": [action(-1., 1., channels=[]),
        action(-.5, .5, channels=[]), action(.75, .25, channels=[])]}, lab)
    source._execute("write", {"path": "parent.py", "code": PARENT}, lab)
    source._execute("evaluate", {"model": "incumbent.py",
        "code": "def predict(row): return row['features']['u']\n"}, lab)
    return_region(source, lab, 0., .5)
    local = return_local_checks(source, lab, expected=2.)
    source._execute("run", {"operator": "bind_research", "model": "parent.py",
        "inputs": {"slope": "slope_question"}}, lab)
    old_fit = source._execute("run", {"operator": "refit", "model": "parent.py",
        "validation_records": [2], "path": "shared/old_fit"}, lab)
    assert old_fit["validation"]["research_bindings"][0]["actual_parameters"]["slope"] == .25
    # New checked child version improves this toy candidate but remains wrong.
    # The outer parent must retain its better incumbent and stay unconfirmed.
    return_region(source, lab, .4, .6)
    source._prepare_program_state()
    checkpoint = source.workspace / "training.json"
    checkpoint.write_text(json.dumps(source.result("model_call_budget_exhausted")))
    (source.workspace / "research_context.json").write_text(json.dumps(source._request(lab)))
    original_checkpoint = checkpoint.read_bytes()
    previous = json.loads(original_checkpoint)

    def make_author(directory):
        return SourceProcedureRepairLearner(None, directory, knowledge=deepcopy(knowledge),
            cases={"same_scene": {"confirmation_seed": 173}}, lab_factory=lambda _: LawLab(),
            config=replace(limits, max_model_calls=8, token_budget=1000), practice_limits=limits,
            prepare_trial=paid_scene_trial_preparer(knowledge, checkpoint, lambda _: LawLab()))

    author = make_author(tmp_path / "author")
    assignment = deepcopy(author.research_root["assignment"])
    root_id = assignment["id"]
    selected = "shared/evidence_revision.py"
    program = procedure_source({"slope": "slope_question", "contrast": {
        "revision_index": local["index"], "check_ids": ["difference"]}}).replace(PROGRAM, METHOD_PATH)
    program = program.replace("if r['path'] ==",
        "if r.get('feedback', {}).get('reason') != 'old_monolith_stop' and r['path'] ==")

    class Provider:
        def __init__(self):
            self.requests = []

        def next_action(self, request):
            self.requests.append(deepcopy(request))
            index = len(self.requests) - 1
            assert request["research_root"]["assignment"] == assignment
            if index == 0:
                proposal = ("run", {"operator": "procedure", "path": METHOD_PATH,
                    "question": {"id": "old_review", "question": "Review the old stopped program."}})
            elif index == 1:
                assert request["research_work_result"]["status"] == "method_stopped"
                proposal = ("finish", {"inconclusive": True, "lesson": "Old program still stops."})
            elif index == 2:
                root = request["research_root"]
                assert request["active_research_question"] == root_id
                assert root["status"] == "open" and root["phase"] == "choose_or_revise_program"
                assert root["latest_child_return"]["question_id"] == "old_review"
                assert root["latest_child_return"]["result"]["directory"] == "trials/trial_000"
                assert author.source_repair is None
                assert json.loads(author.files["method_candidates.json"])[0]["source_path"].startswith("trials/")
                # A second unscoped local finish cannot silently end the wider
                # assignment after the first local return has cleared focus.
                proposal = ("finish", {"inconclusive": True, "lesson": "The same old rejection."})
            elif index == 3:
                assert author.events[-1]["feedback"]["status"] == "source_repair_pending"
                assert request["research_root"]["status"] == "open"
                proposal = ("run", {"code": "from pathlib import Path\n" +
                    f"Path({selected!r}).write_text({program!r})"})
            elif index == 4:
                proposal = ("run", {"operator": "procedure", "path": selected,
                    "question": {"id": "evidence_revision", "question": "Use revised local evidence in the fit.",
                                 "depends_on": ["slope_question", "local"]}})
            elif index == 5:
                operations = request["research_work_result"]["operations"]
                assert [item["operator"] for item in operations[:2]] == ["bind_research", "refit"]
                assert operations[1]["executed_bindings"][0]["inputs"]["slope"]["version"] == 2
                assert operations[1]["executed_bindings"][0]["actual_parameters"]["slope"] == .5
                assert operations[2]["retention_decision"]["status"] == "incumbent_retained"
                proposal = ("finish", {"inconclusive": True, "lesson": "Revised candidate remains worse."})
            else:
                assert index == 6
                assert request["research_root"]["status"] == "open"
                assert request["active_research_question"] == root_id
                assert request["research_root"]["latest_child_return"]["question_id"] == "evidence_revision"
                proposal = ("finish", {"inconclusive": True, "question_id": root_id,
                    "lesson": "Keep the incumbent; the fixture leaves the parent unresolved."})
            return {"action": proposal[0], "arguments": proposal[1], "usage": {"total_tokens": 5}}

    provider = author.provider = Provider()
    result = author.run()
    assert len(provider.requests) == 7 and len(author.practice_trials) == 2
    assert result["status"] == "source_repair_inconclusive"
    assert result["published"] is False and result["model"] is None
    assert result["research_root"]["assignment"] == assignment
    assert result["costs"]["observation_cost"] == 0
    assert result["research_work_result"]["confirmation"] is None
    trial = author.workspace / author.practice_trials[-1]["directory"]
    saved = json.loads((trial / "result.json").read_text())
    assert saved["paid_observations"] == previous["paid_observations"]
    assert saved["evaluated_model"]["mse"] == previous["evaluated_model"]["mse"] == 0.
    fit = result["research_work_result"]["operations"][1]
    assert 0. < fit["validation"]["mse"] < old_fit["validation"]["mse"]
    assert checkpoint.read_bytes() == original_checkpoint
    assert (source.workspace / METHOD_PATH).read_text() == OLD_METHOD

    resumed = make_author(tmp_path / "resumed")
    resumed.restore_practice(result)
    request = resumed._request(resumed.interface)
    assert request["research_root"]["assignment"] == assignment
    assert request["research_root"]["latest_child_return"]["question_id"] == "evidence_revision"
    assert request["active_research_question"] == root_id
    assert resumed._terminal_status() is None
    assert len(resumed.practice_trials) == 2 and resumed.costs["observation_cost"] == 0
    assert all((resumed.workspace / item["directory"]).is_symlink() for item in resumed.practice_trials)

    original_child = deepcopy(next(q for q in resumed.questions if q["id"] == "evidence_revision"))
    resumed.config = replace(resumed.config, observation_budget=0)
    denied = resumed._execute("run", {"operator": "procedure", "path": METHOD_PATH,
        "question": {"id": "evidence_revision", "question": "A revised local working hypothesis."}},
        resumed.interface)
    assert "error" in denied and len(resumed.practice_trials) == 2
    child = next(q for q in resumed.questions if q["id"] == "evidence_revision")
    assert child["assignment"] == original_child["assignment"]
    assert child["question"] != child["assignment"]["question"]
    assert resumed.research_root["assignment"] == assignment
