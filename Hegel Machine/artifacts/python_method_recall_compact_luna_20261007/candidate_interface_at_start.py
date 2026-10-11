"""Learn one executable method from paid, unlabeled paired situations."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

from .code_provider_context import _compact_trajectory_payload
from .online_inquiry_llm import InquiryLLMAgent
from .temporal_method_learning import METHOD_PATH
from .temporal_paired_method import FrozenMethodAgent
from .temporal_operator_agent import OPERATOR_INTERFACE


PAIRED_INTERFACE = """
Learn a self-contained research program that checks whether its earlier supporting
model still applies, preserves useful knowledge, and repairs it when evidence
requires that. The listed practice cases expose the same public interface and earlier
knowledge. Their identities do not reveal whether anything changed. Choose your
own experiments, numerical conditions and repairs. Do not always rebuild first:
the question is whether your program can decide from evidence when that is useful.
You can revise the inherited program before any execution; it is a reference,
not a mandatory first experiment.

AUTHORING ACTIONS:
write {path:'shared/consolidated_method.py',code:'Python source'};
run {code:'Python source'} to inspect your files and paid practice traces;
run {operator:'practice',case:CASE_LABEL} to execute the current program
from empty observations and the SAME frozen prior knowledge in a fresh workspace;
run {operator:'restore_method',trial:'trials/trial_NNN'} to select a previously
practiced source exactly as saved in that trial;
finish {path:'shared/consolidated_method.py',lesson:'...'} to freeze your method.
Each practice costs real observations, measurements and computation. Feedback
provides the relative trial directory; its result.json and paid evidence files
are available for inspection. No automatic refitting occurs before your program.
method_candidates.json lists your practiced sources, grouped by exact source,
with confirmed and failed cases, actual trial directories, and the current source.
The same case can appear in both lists when that source had different outcomes.
Choose a candidate's trial to restore its source and existing same-source coverage.
Restoration does not publish or repair the program; missing cases still require
practice under the same publication conditions. You may also keep editing.
Any study-specific practice-after-edit rule also applies to restoring a source.
Publication needs fresh predictive confirmation on EVERY listed case with exactly the
current source. It must actually use the unchanged earlier predictor on at least
one case, and confirm a revised predictor on another case. Which case supports
which choice is not supplied: establish that from paid evidence. Solving all cases
by refitting does not demonstrate the requested reuse-and-revision capability.
A final unexecuted edit loses that coverage. Retesting any case
starts from its original prior, not the previous trial's fitted numerical models.
Write and practice feedback report current_source_coverage: the cases that actually
confirmed this exact program, the cases still missing, and its observed reuse and
revision outcomes. This is evidence about the current file, not earlier versions.
If resume_context.json is present, an earlier authoring phase ended before
publication. That file records its status, saved current-source coverage and last action; the
underlying paid trial files remain available. Publication conditions are unchanged.
Ordinary Python inspection is available; query/evaluate and individual fitting
operators are not authoring actions. Request experiments through your program.

EXECUTABLE PROGRAM CONTRACT:
def select_actions(public_spec, records, memory): ...
records is a list of trajectory RECORDS, each with action, rows, measurements and
row_indices. len(records) counts trajectories, NOT observation rows. Existing
record indices are range(len(records)); observation rows count as
sum(len(record['rows']) for record in records). No records exist at the first call.
memory['research_roles'] gives instrument and predictor file paths for the earlier
knowledge. Use these roles across apparatuses; do not infer mechanisms from case
identifiers, filenames or directories. Newly fitted files are local to a trial.
research_state.json exposes flat operator_results and method_results from actual
previous invocations, plus costs and limits. All are reset for a new trial.
Practice feedback.steps uses these same method_results entries, each containing
{step,path,proposal,feedback}; a submission's feedback
contains fresh_confirmation:{passed,mse,rows,...} or an error. The last operator
result has operator and its numerical feedback fields at the same level.

Return {'actions':[trajectory,...], 'after_observation':[operator arguments,...]}
to buy data and analyze it, or actions:[] with nonempty after_observation to analyze
already-paid records. A whole continuation runs linearly; the next invocation can
branch on its actual outcomes. Both lists empty is invalid.
Return {'model':'path/to/predictor.py','lesson':'...'} alone to request fresh
confirmation and finish. Selecting the unchanged earlier predictor is allowed.
The host never selects a candidate or an experimental strategy for you. A failed
confirmation adds its counterexamples to paid records and permits another step.
The exact same source executes up to eight steps per trial, without another LLM
choosing intermediate actions. After publication it will also run from empty
observations with another previously learned calibration. Only your source and
lesson transfer; practice-fitted models and observations do not.

OPERATOR REFERENCE FOR THE PROGRAM'S after_observation ENTRIES:
""" + OPERATOR_INTERFACE


class _PracticeInterface:
    def __init__(self, spec, cases):
        self.spec = deepcopy(spec)
        self.task_interface = ("Author and practice a research program on these cases: "
                               + ", ".join(cases) + ".\n")

    def public_spec(self):
        return deepcopy(self.spec)


class PairedMethodLearner(InquiryLLMAgent):
    common_interface = PAIRED_INTERFACE

    def __init__(self, provider, workspace, *, knowledge, cases, lab_factory, config):
        super().__init__(provider, workspace, arm="hegel", knowledge=knowledge,
                         config=config, source_task=False)
        self.cases, self.lab_factory = deepcopy(cases), lab_factory
        self.practice_trials = []
        self.published_source = None
        self.common_interface = PAIRED_INTERFACE.replace("CASE_LABEL", repr(next(iter(cases))))
        self.interface = _PracticeInterface(lab_factory(next(iter(cases.values()))).public_spec(), cases)
        self._refresh_method_candidates()

    def run(self, *, on_checkpoint=None):
        return super().run(self.interface, on_checkpoint=on_checkpoint)

    def restore_practice(self, previous_result):
        """Retain paid trials from an unfinished phase; charge new work separately."""
        if previous_result["status"] not in {
                "provider_failed", "token_budget_exhausted", "model_call_budget_exhausted"}:
            raise ValueError("practice restoration expects an unpublished interrupted or budget-exhausted result")
        source = previous_result["outgoing_knowledge"]["files"][METHOD_PATH]
        (self.workspace / METHOD_PATH).write_text(source)
        self.files[METHOD_PATH] = source
        self.practice_trials = deepcopy(previous_result["practice_trials"])
        previous_workspace = Path(previous_result["workspace"])
        for trial in self.practice_trials:
            shutil.copytree(previous_workspace / trial["directory"], self.workspace / trial["directory"])
        last_action = next((event for event in reversed(previous_result["events"])
                            if event["action"] != "provider_error"), None)
        last_action_summary = _compact_trajectory_payload(last_action)
        if last_action is not None:
            action_file = "resume_last_action.json"
            (self.workspace / action_file).write_text(json.dumps(last_action, indent=2))
            last_action_summary["full_action_file"] = action_file
            if last_action["action"] == "run" and "operator" not in last_action.get("arguments", {}):
                feedback = last_action_summary.get("feedback", {})
                stdout = feedback.get("stdout", "")
                if len(stdout) > 2000:
                    # A previous inspection's output must not crowd the resume
                    # status and paid-practice record out of an ordinary read.
                    feedback.update(stdout=stdout[:256], stdout_characters=len(stdout),
                                    stdout_truncated=True)
                    last_action_summary.pop("usage", None)
        context = {"resumed_from": str(previous_workspace), "previous_status": previous_result["status"],
                   "current_source_coverage": self._current_source_coverage(),
                   "previous_practice": [{key: value for key, value in trial.items() if key != "method_source"}
                                         for trial in self.practice_trials],
                   "last_action": last_action_summary,
                   "cost_scope": "Earlier paid trials are retained. Current costs and limits apply only "
                                 "to this new authoring phase; earlier costs remain in the preceding result."}
        self.files["resume_context.json"] = json.dumps(context, indent=2)
        (self.workspace / "resume_context.json").write_text(self.files["resume_context.json"])
        return context

    def _refresh_method_candidates(self):
        """Expose the source snapshots and paid outcomes already held by practice."""
        current = (self.workspace / METHOD_PATH).read_text() if self.practice_trials else None
        grouped = {}
        for trial in self.practice_trials:
            source = trial["method_source"]
            if source not in grouped:
                grouped[source] = {"trial": trial["directory"],
                    "source_path": str(Path(trial["directory"]) / METHOD_PATH),
                    "current": source == current, "confirmed_cases": [], "failed_cases": [],
                    "trials": []}
            candidate = grouped[source]
            cases = candidate["confirmed_cases" if trial["status"] == "completed" else "failed_cases"]
            if trial["case"] not in cases:
                cases.append(trial["case"])
                cases.sort()
            candidate["trials"].append({key: deepcopy(trial[key]) for key in (
                "case", "directory", "status", "fresh_confirmation", "selected_prior_predictor")})
        content = json.dumps(list(grouped.values()), indent=2)
        self.files["method_candidates.json"] = content
        (self.workspace / "method_candidates.json").write_text(content)

    def _current_source_coverage(self):
        source = (self.workspace / METHOD_PATH).read_text()
        successful = [trial for trial in self.practice_trials
                      if trial["method_source"] == source and trial["status"] == "completed"]
        confirmed = {trial["case"] for trial in successful}
        coverage = {"confirmed_cases": sorted(confirmed),
                "missing_cases": sorted(set(self.cases) - confirmed),
                "prior_reuse_cases": sorted({trial["case"] for trial in successful
                                             if trial["selected_prior_predictor"]}),
                "revised_cases": sorted({trial["case"] for trial in successful
                                         if not trial["selected_prior_predictor"]})}
        if "resume_context.json" in self.files:
            context = json.loads(self.files["resume_context.json"])
            context["current_source_coverage"] = coverage
            self.files["resume_context.json"] = json.dumps(context, indent=2)
            (self.workspace / "resume_context.json").write_text(self.files["resume_context.json"])
        self._refresh_method_candidates()
        return coverage

    def _execute(self, action, arguments, laboratory):
        if action == "write":
            if arguments.get("path") != METHOD_PATH:
                raise ValueError("write the self-contained method at shared/consolidated_method.py")
            feedback = super()._execute(action, arguments, laboratory)
            return {**feedback, "current_source_coverage": self._current_source_coverage()}
        if action == "run" and "operator" not in arguments:
            feedback = super()._execute(action, arguments, laboratory)
            source = (self.workspace / METHOD_PATH).read_text()
            if source != self.files[METHOD_PATH]:
                # Ordinary Python can adopt or edit a stored method just as write can.
                # The next author decision must see the source that practice executes.
                self.files[METHOD_PATH] = source
                return {**feedback, "current_source_coverage": self._current_source_coverage()}
            return feedback
        if action == "run" and arguments.get("operator") == "practice":
            return self._practice(arguments["case"])
        if action == "run" and arguments.get("operator") == "restore_method":
            trial = next((item for item in self.practice_trials
                          if item["directory"] == arguments.get("trial")), None)
            if trial is None:
                raise ValueError("choose a recorded trial from method_candidates.json")
            feedback = super()._execute("write", {"path": METHOD_PATH,
                "code": trial["method_source"]}, laboratory)
            return {**feedback, "restored_method": trial["directory"],
                    "current_source_coverage": self._current_source_coverage()}
        if action != "finish":
            raise ValueError("authoring uses write, ordinary run, run operator:practice/restore_method, and finish")
        if arguments.get("path") != METHOD_PATH:
            raise ValueError("finish needs path:shared/consolidated_method.py")
        source = (self.workspace / METHOD_PATH).read_text()
        coverage = self._current_source_coverage()
        confirmed, missing = coverage["confirmed_cases"], coverage["missing_cases"]
        if missing:
            return {"error": "the current method still needs fresh confirmation on: " + ", ".join(missing),
                    "unconfirmed_cases": missing}
        reused, revised = coverage["prior_reuse_cases"], coverage["revised_cases"]
        if not any(a != b for a in reused for b in revised):
            return {"error": "prediction succeeded but the current method has not demonstrated actual "
                             "unchanged-prior reuse in one case and confirmed revision in another; "
                             "use evidence to decide rather than rebuilding in every case",
                    "confirmed_prior_reuse_cases": sorted(reused), "confirmed_revision_cases": sorted(revised)}
        additional_error = self._publication_error(coverage)
        if additional_error is not None:
            return additional_error
        self.model, self.lesson, self.published_source = METHOD_PATH, arguments.get("lesson", ""), source
        return {"published": METHOD_PATH, "confirmed_cases": sorted(confirmed)}

    def _publication_error(self, coverage):
        return None

    def _practice(self, case):
        if case not in self.cases:
            raise ValueError("choose " + " or ".join(self.cases))
        remaining = self.config.observation_budget - self.costs["observation_cost"]
        if remaining < 48:
            return {"error": "remaining observations cannot cover fresh confirmation"}
        source = (self.workspace / METHOD_PATH).read_text()
        directory = f"trials/trial_{len(self.practice_trials):03d}"
        limits = replace(self.config, observation_budget=min(240, remaining), confirmation_rows=48,
            compute_budget=min(100000, self.config.compute_budget - self.costs["compute_cost"]),
            execution_seconds=min(60., self.config.execution_seconds - self.costs["python_execution_seconds"]))
        agent = FrozenMethodAgent(self.workspace / directory, knowledge=self.knowledge,
            method_source=source, config=limits,
            confirmation_seed=self.cases[case]["confirmation_seed"] + len(self.practice_trials))
        result = agent.execute_method(self.lab_factory(self.cases[case]), max_steps=8)
        prior_path = self.knowledge["research_roles"]["predictor"]
        selected_prior = (result["model"] == prior_path
                          and (agent.workspace / prior_path).read_text() == self.knowledge["files"][prior_path])
        return self._record_practice(case, directory, source, agent, result, selected_prior)

    def _record_practice(self, case, directory, source, agent, result, selected_prior):
        """Keep paid outcomes and current-source coverage shared across laboratories."""
        (agent.workspace / "result.json").write_text(json.dumps(result, indent=2))
        for key in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
            self.costs[key] += result["costs"][key]
        confirmation = result.get("fresh_confirmation")
        feedback = {"case": case, "directory": directory, "status": result["status"],
                    "costs": result["costs"], "model": result["model"],
                    "selected_prior_predictor": selected_prior,
                    "fresh_confirmation": {key: confirmation.get(key) for key in ("passed", "mse", "rows")}
                        if confirmation else None,
                    "steps": deepcopy(result["method_results"])}
        self.practice_trials.append({"case": case, "directory": directory,
            "method_source": source, "status": result["status"], "costs": result["costs"],
            "selected_prior_predictor": selected_prior,
            "fresh_confirmation": feedback["fresh_confirmation"]})
        feedback["current_source_coverage"] = self._current_source_coverage()
        return feedback

    def result(self, status):
        result = super().result(status)
        result.update(practice_trials=deepcopy(self.practice_trials),
                      method_source=self.published_source, method_path=self.model, lesson=self.lesson,
                      method_entry={"id": "paired_evidence_method", "path": METHOD_PATH}
                          if self.published_source else None)
        return result
