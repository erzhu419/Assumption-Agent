"""Practice a reusable procedure that changes its steps with paid feedback."""

from .temporal_method_learning import METHOD_PATH
from .temporal_method_practice import MethodPracticeAgent


CONDITIONAL_INTERFACE = """
This practice asks you to retain evidence-dependent decisions INSIDE the learned
research program. A reusable procedure should examine its task's observations and
actual prior analysis results in research_state.json, then choose its next useful
steps. Rewriting the file after every result can solve this task without retaining
those decisions. Keep the decisions executable so the same final source can take
different research steps when evidence changes. No decision thresholds, acquisition
schedule or fitting policy are supplied; devise and test your own.

Before publication, execute the same final revised source at least twice with
different substantive stages (acquisition versus analysis, or different analysis
operators), with intervening operator feedback. Merely changing question text,
output paths or repeating the same operator sequence does not demonstrate this.
Use substantive evidence or analysis outcomes for decisions, not an invocation
counter. The current program must still produce the freshly confirmed predictor.
These execution requirements establish a minimal trace; the actual source and
feedback determine whether it learned a meaningful condition.
"""


class ConditionalMethodPracticeAgent(MethodPracticeAgent):
    common_interface = CONDITIONAL_INTERFACE + MethodPracticeAgent.common_interface

    def conditional_trials(self):
        source = (self.workspace / METHOD_PATH).read_text(encoding="utf-8")
        trials = [trial for trial in self.method_trials if trial["method_source"] == source
                  and not trial.get("error") and not trial.get("after_observation_error")]
        # Same-source executions must change an actual stage after more feedback.
        # This is execution evidence, not a classifier of semantic understanding.
        for earlier in trials:
            for later in trials:
                if (later["trial_index"] > earlier["trial_index"]
                        and later["operator_count_before"] > earlier["operator_count_before"]
                        and self._stage(later) != self._stage(earlier)):
                    return [trial["trial_index"] for trial in trials]
        return []

    @staticmethod
    def _stage(trial):
        return (bool(trial["record_indices"]), tuple(trial["operator_sequence"]))

    def _eligible_trial(self, model):
        eligible = super()._eligible_trial(model)
        return eligible if self.conditional_trials() else None

    def _execute(self, action, arguments, laboratory):
        if (action == "finish" and super()._eligible_trial(arguments.get("model", "model.py")) is not None
                and not self.conditional_trials()):
            return {"error": "the current revised source has not executed different substantive research "
                             "stages in response to intervening operator feedback; retain the decisions "
                             "inside the program and execute it unchanged again"}
        return super()._execute(action, arguments, laboratory)

    def result(self, status):
        result = super().result(status)
        result["conditional_method_trials"] = self.conditional_trials()
        return result
