"""A source author's selected program executes evidence-to-comparison as one work.

This uses only a synthetic affine fixture, never a scientific live candidate.
"""

from copy import deepcopy
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
from test_temporal_python_refit import LawLab


OLD_METHOD = "def select_actions(spec, observations, memory):\n return {'stop':'old_monolith_stop'}\n"


def test_source_author_selection_executes_current_evidence_refit_and_comparison(tmp_path):
    knowledge = {"files": {METHOD_PATH: OLD_METHOD, "shared/knowledge.json":
        json.dumps({"hypotheses": [], "research_programs": [], "support_questions": []})}}
    config = InquiryLLMConfig(observation_budget=60, compute_budget=10000,
        confirmation_rows=3, execution_seconds=45., max_model_calls=0, token_budget=0)
    lab = LawLab()
    source = CommonArchiveInquiryAgent(None, tmp_path / "source", arm="hegel",
        source_task=False, knowledge=deepcopy(knowledge), config=config)
    source._execute("query", {"actions": [action(-1., 1., channels=[]),
        action(-.5, .5, channels=[]), action(.75, .25, channels=[])]}, lab)
    source._execute("write", {"path": "parent.py", "code": PARENT}, lab)
    source._execute("evaluate", {"model": "incumbent.py",
        "code": "def predict(row): return row['features']['u']\n"}, lab)
    return_region(source, lab, 0., .5)
    local = return_local_checks(source, lab, expected=2.)
    source._execute("run", {"operator": "bind_research", "model": "parent.py",
                           "inputs": {"slope": "slope_question"}}, lab)
    return_region(source, lab, .9, 1.1)
    source._prepare_program_state()
    previous = source.result("model_call_budget_exhausted")
    checkpoint = source.workspace / "training.json"
    checkpoint.write_text(json.dumps(previous))
    previous = json.loads(checkpoint.read_text())
    (source.workspace / "research_context.json").write_text(json.dumps(source._request(lab)))
    original_checkpoint = checkpoint.read_bytes()
    original_source = (source.workspace / METHOD_PATH).read_text()

    author = SourceProcedureRepairLearner(None, tmp_path / "author", knowledge=knowledge,
        cases={"same_scene": {"confirmation_seed": 173}}, lab_factory=lambda _: LawLab(),
        config=config, practice_limits=config,
        prepare_trial=paid_scene_trial_preparer(knowledge, checkpoint, lambda _: LawLab()))
    stopped = author._practice("same_scene")
    assert stopped["status"] == "method_stopped"
    assert stopped["continuation_cost"]["observation_cost"] == 0

    selected = "shared/selected_research.py"
    program = procedure_source({"slope": "slope_question", "contrast": {
        "revision_index": local["index"], "check_ids": ["difference"]}}).replace(PROGRAM, METHOD_PATH)
    # The authored program follows its own sequence, not the earlier monolith's
    # terminal step. It keeps the established source interface and action names.
    program = program.replace("if r['path'] ==", "if r.get('feedback', {}).get('reason') != 'old_monolith_stop' and r['path'] ==")
    author._execute("run", {"code": f"from pathlib import Path\nPath({selected!r}).write_text({program!r})"},
                    author.interface)
    question = {"id": "work", "question": "Use the selected local evidence in the same paid-scene comparison.",
                "parent_id": "task", "depends_on": ["slope_question", "local"]}
    result = author._execute("run", {"operator": "procedure", "path": selected,
                                    "question": question}, author.interface)

    assert result["status"] == "method_stopped"
    assert result["continuation_cost"]["observation_cost"] == 0
    assert result["fresh_confirmation"] is None
    work = result["research_work_result"]
    assert work["confirmation"] is None
    assert work["declared_binding_models"] == ["parent.py"]
    operations = work["operations"]
    assert [item["operator"] for item in operations[:2]] == ["bind_research", "refit"]
    assert [item["action"] for item in operations] == ["run", "run", "evaluate", None]
    fit = operations[1]
    assert fit["validation"]["mse"] == 0.
    assert fit["split"]["validation_records_count"] == 1
    consumed = fit["executed_bindings"][0]
    assert consumed["inputs"]["slope"]["version"] == 2
    assert consumed["inputs"]["contrast"]["revision_index"] == local["index"]
    assert consumed["actual_parameters"] == {"slope": 1.}
    assert operations[2]["mse"] == 0.
    assert operations[2]["retention_decision"]["status"] == "incumbent_retained"
    assert operations[0]["executed_bindings"] == []  # Declaration alone is not consumption.
    assigned = author.research_work["question"]
    assert assigned["parent_id"] == author.research_root["id"]
    assert assigned["assignment"] == {**question, "parent_id": author.research_root["id"]}
    assert author.research_work["selected_source"] == selected
    assert (author.workspace / METHOD_PATH).read_text() == program
    assert (author.workspace / selected).read_text() == program
    assert checkpoint.read_bytes() == original_checkpoint
    assert (source.workspace / METHOD_PATH).read_text() == original_source == OLD_METHOD
    trial = author.workspace / result["directory"]
    saved = json.loads((trial / "result.json").read_text())
    actual_fit = saved
    for key in fit["evidence"]["json_path"]:
        actual_fit = actual_fit[key]
    assert actual_fit['split']['validation_records'] == [2]
    assert actual_fit['validation']['research_bindings'][0]['actual_parameters'][
        'training_inputs'] == [-1., 1., -.5, .5]
    assert saved["paid_observations"] == previous["paid_observations"]
    assert saved["research_work"] == author.research_work
    assert saved["evaluated_model"]["mse"] == previous["evaluated_model"]["mse"]
    assert next(q for q in saved["online_questions"] if q["id"] == "work")["depends_on"] == question["depends_on"]
    assert author.costs["model_calls"] == author.costs["observation_cost"] == 0
    assert not author.source_repair and not author.published_source
    assert "error" in author._execute("finish", {"path": METHOD_PATH}, author.interface)
