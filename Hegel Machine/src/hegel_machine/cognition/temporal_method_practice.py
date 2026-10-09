"""Practice a model-authored acquisition-and-analysis method on a real task."""

from copy import deepcopy
import json

from .online_inquiry_llm import COMMON_INTERFACE
from .temporal_method_learning import METHOD_PATH
from .temporal_operator_agent import TemporalOperatorAgent


PRACTICE_INTERFACE = """
Your goal in this task is to improve an executable research method through actual
practice, then confirm a predictor produced by that method. The inherited seed
procedure is shared/consolidated_method.py. Begin by executing that unchanged
procedure on the real laboratory. Examine its purchased evidence and feedback,
revise the procedure, and execute the changed version on paid evidence.
Choose the research strategy yourself; no experiment schedule is supplied.

All research acquisitions must use query {path:'shared/consolidated_method.py',
question:{id:'...',question:'...',parent_id:null,depends_on:[]}}. Put your analysis
steps in the procedure's returned after_observation list, not the query arguments.
Each continuation executes linearly after its acquisition; branching on its
results requires a later procedure invocation. memory is the inherited snapshot,
observations contains the current task's paid laboratory records, and instrument
files on disk contain the latest fitted or assessed state. research_state.json
contains actual completed operator results and the current costs and limits.
A program can return actions:[] with nonempty after_observation to analyze
already-paid records; this uses computation but buys no observations.
The public laboratory
interface describes controls and trajectories; it does not contain budget fields.
The limits and costs in the agent request specify your task budget.
Each paid trial reports whether the program actually changed since its previous
trial and how many acquired actions exactly repeat earlier actions, including
their channels. These counts describe execution, not estimated information gain;
a changed question description alone does not change the experiment.

You may use write, ordinary run, the shared run operators, and evaluate to inspect
or explore. For finish, the submitted predictor must be an unmodified export of
a successful history or response step in the CURRENT procedure's own continuation.
There must be an earlier paid trial and a later paid trial of a changed procedure.
A final edit that has never run on paid evidence cannot supply that predictor.
Use the ordinary Hegel manifest with this procedure in research_programs and the
selected predictor in hypotheses; finish still buys fresh predictive confirmation.
An analysis result or a method edit alone does not finish the task.

Later transfer will restore the original predictive knowledge and retain only
your practiced procedure and its description/lesson. Instrument calibrations,
predictors, auxiliary files and observations newly generated in this practice
workspace will not transfer. Make the procedure self-contained and able to use
its new task's observations; create any new analysis files through its own steps.
"""


class MethodPracticeAgent(TemporalOperatorAgent):
    """Keep acquisition, method revision, and its actual model exports connected."""

    common_interface = PRACTICE_INTERFACE + COMMON_INTERFACE

    def __init__(self, provider, workspace, *, knowledge, config=None, confirmation_seed=9173):
        super().__init__(provider, workspace, arm="hegel", knowledge=knowledge, config=config,
                         confirmation_seed=confirmation_seed, source_task=False)
        if METHOD_PATH not in self.files:
            raise ValueError("method practice requires an inherited seed procedure")
        self.seed_method_source = self.files[METHOD_PATH]
        self.method_trials = []
        self.confirmed_method_trial = None

    def _query(self, arguments, laboratory):
        if arguments.get("path") != METHOD_PATH:
            return {"error": "practice acquisitions must execute shared/consolidated_method.py"}
        if "after_observation" in arguments:
            return {"error": "put after_observation inside the research procedure, not query arguments"}
        method_source = (self.workspace / METHOD_PATH).read_text(encoding="utf-8")
        if not self.method_trials and method_source != self.seed_method_source:
            return {"error": "execute the unchanged inherited seed on paid evidence before revising it"}
        before_records = len(self.observations)
        before_cost = self.costs["observation_cost"]
        before_operators = len(self.operator_results)
        result = super()._query(arguments, laboratory)
        # A failed continuation can still have acquired the evidence needed for
        # method revision. Keep such trials instead of counting successes only.
        if (len(self.observations) == before_records
                and not (result.get("continuation_origin") == METHOD_PATH
                         and result.get("after_observation_results"))):
            return result
        seen_actions = [record["action"] for record in self.observations[:before_records]]
        repeated = 0
        for record in self.observations[before_records:]:
            repeated += record["action"] in seen_actions
            seen_actions.append(record["action"])
        trial_feedback = {
            "program_changed_since_previous_trial": (
                method_source != self.method_trials[-1]["method_source"] if self.method_trials else None),
            "new_action_count": len(self.observations) - before_records - repeated,
            "repeated_action_count": repeated,
            "comparison": "Exact acquired actions including channels; not an information-gain estimate.",
        }
        exported = []
        if result.get("continuation_origin") == METHOD_PATH:
            for outcome in result.get("after_observation_results", []):
                if "error" in outcome:
                    continue
                if outcome.get("operator") == "history":
                    paths = [outcome["path"]]
                elif outcome.get("operator") == "response":
                    paths = [candidate["path"] for candidate in outcome.get("candidates", [])]
                else:
                    continue
                for path in paths:
                    exported.append({"path": path, "operator": outcome["operator"],
                                     "source": (self.workspace / path).read_text(encoding="utf-8")})
        trial = {"trial_index": len(self.method_trials), "method_source": method_source,
                 **trial_feedback,
                 "operator_count_before": before_operators,
                 "operator_sequence": [outcome.get("operator")
                     for outcome in result.get("after_observation_results", [])],
                 "record_indices": list(range(before_records, len(self.observations))),
                 "observation_cost": self.costs["observation_cost"] - before_cost,
                 "continuation_origin": result.get("continuation_origin"),
                 "after_observation": deepcopy(result.get("after_observation", [])),
                 "after_observation_results": deepcopy(result.get("after_observation_results", [])),
                 "exported_candidates": exported}
        for key in ("error", "after_observation_error"):
            if key in result:
                trial[key] = result[key]
        self.method_trials.append(trial)
        result["method_trial_index"] = trial["trial_index"]
        result["method_trial_feedback"] = trial_feedback
        return result

    def _eligible_trial(self, model):
        current_source = (self.workspace / METHOD_PATH).read_text(encoding="utf-8")
        if len(self.method_trials) < 2:
            return None
        model_path = self.workspace / model
        if not model_path.is_file():
            return None
        model_source = model_path.read_text(encoding="utf-8")
        for index in range(len(self.method_trials) - 1, 0, -1):
            trial = self.method_trials[index]
            if trial["method_source"] != current_source:
                continue
            if not any(previous["method_source"] != current_source for previous in self.method_trials[:index]):
                continue
            if any(candidate["path"] == model and candidate["source"] == model_source
                   for candidate in trial["exported_candidates"]):
                return index
        return None

    def _execute(self, action, arguments, laboratory):
        if (action == "write" and arguments.get("path") == METHOD_PATH
                and not self.method_trials and arguments.get("code") != self.seed_method_source):
            return {"error": "execute the unchanged inherited seed on paid evidence before revising it"}
        if action != "finish":
            return super()._execute(action, arguments, laboratory)
        model = arguments.get("model", "model.py")
        trial_index = self._eligible_trial(model)
        if trial_index is None:
            return {"error": "finish requires an unmodified predictor exported by the current procedure's own "
                             "history/response continuation after a paid trial of an earlier procedure version"}
        manifest_path = self.workspace / arguments.get("manifest", "shared/knowledge.json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if not any(entry.get("path") == METHOD_PATH for entry in manifest.get("research_programs", [])):
            return {"error": "the manifest must retain the practiced procedure in research_programs"}
        result = super()._execute(action, arguments, laboratory)
        if self.model:
            self.confirmed_method_trial = trial_index
            result["confirmed_method_trial"] = trial_index
        return result

    def result(self, status):
        result = super().result(status)
        published = status == "completed" and self.model and self.confirmed_method_trial is not None
        entry = next((entry for entry in self.manifest.get("research_programs", [])
                      if entry.get("path") == METHOD_PATH), None) if published else None
        result.update(method_trials=deepcopy(self.method_trials),
                      confirmed_method_trial=self.confirmed_method_trial,
                      method_path=METHOD_PATH if entry else None,
                      method_source=(self.workspace / METHOD_PATH).read_text(encoding="utf-8") if entry else None,
                      method_entry=deepcopy(entry), lesson=self.lesson)
        return result
