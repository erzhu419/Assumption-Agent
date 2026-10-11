"""Return one real counterexample practice after an author's source edit."""

from hegel_machine.cognition.code_provider_context import _compact_practice_feedback
from hegel_machine.cognition.python_scoped_method_learning import ScopedPythonMethodLearner
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH


EDIT_PRACTICE_INTERFACE = """
In this study, changing shared/consolidated_method.py through write or ordinary
run also requests one paid practice on the supplied counterexample case
{counterexample_case}. The host executes your changed program with the existing
tools, fresh-confirmation stream and remaining author budget, then returns the
actual outcome under edit_practice. This includes execution failures and unmet
confirmation, so use that feedback for your next revision. Pure inspection and
writing unchanged source do not buy another practice. A case already confirmed
for exactly the current source retains that coverage without being purchased again.
This automatic practice never repairs your source or publishes it. You still
choose further edits or explicit practice and finish; the existing finish action
executes all remaining cases and enforces the unchanged publication requirements.
"""


class EditPracticeMethodLearner(ScopedPythonMethodLearner):
    """Reuse the normal paid practice and publication logic after source changes."""

    def __init__(self, *args, counterexample_case, **kwargs):
        super().__init__(*args, **kwargs)
        if counterexample_case not in self.cases:
            raise ValueError("the counterexample must name a supplied practice case")
        self.counterexample_case = counterexample_case
        self.common_interface = self.common_interface.replace(
            "Writing alone never starts a practice.", "")
        self.common_interface += EDIT_PRACTICE_INTERFACE.format(
            counterexample_case=counterexample_case)

    def _execute(self, action, arguments, laboratory):
        path = self.workspace / METHOD_PATH
        before = path.read_text()
        feedback = super()._execute(action, arguments, laboratory)
        if path.read_text() == before:
            return feedback
        coverage = self._current_source_coverage()
        if self.counterexample_case not in coverage["missing_cases"]:
            return feedback
        outcome = self._practice(self.counterexample_case)
        return {**feedback, "edit_practice": _compact_practice_feedback(outcome),
                "current_source_coverage": self._current_source_coverage()}
