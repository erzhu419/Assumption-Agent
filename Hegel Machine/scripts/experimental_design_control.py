"""Matched experiment-selection controls for the frozen authored research method."""

from copy import deepcopy
from random import Random

from hegel_machine.cognition.anonymous_method_learning import SingleSubmissionMethodAgent


class ExperimentalDesignMethodAgent(SingleSubmissionMethodAgent):
    """Allow sequential proposal, comparison, acquisition and readout before submission."""

    def execute_method(self, laboratory, max_steps=32, *, on_checkpoint=None):
        return super().execute_method(laboratory, max_steps=max(32 - len(self.method_results), max_steps),
                                      on_checkpoint=on_checkpoint)


def _replace_selection(agent, result, choices, chosen, control):
    """Replace a choice after the shared comparison, preserving its paid evidence."""
    record = agent.experiment_comparisons[result['comparison']['index']]
    original = record['selected_experiment']
    selected = {key: deepcopy(chosen[key]) for key in
                ('id', 'actions', 'observation_cost', 'predicted_separation')}
    selected['revision'] = deepcopy(original['revision'])
    selected['revision']['id'] = f"comparison_{record['index']}_{chosen['id']}"
    record['experiment_selection_control'] = {
        **control,
        'eligible_experiments': [experiment['id'] for experiment in choices],
        'greedy_suggestion': deepcopy(original),
        'selected_experiment': chosen['id'],
    }
    record['selected_experiment'] = selected
    record['query_arguments'] = {**deepcopy(record['query_arguments']),
        'actions': deepcopy(selected['actions']),
        'revision': deepcopy(selected['revision'])}
    agent._prepare_program_state()
    return {**result, 'comparison': deepcopy(record)}


class RandomExperimentMethodAgent(ExperimentalDesignMethodAgent):
    """Replace only the comparison's selection with a same-cost random experiment."""

    def __init__(self, *args, confirmation_seed=9173, **kwargs):
        super().__init__(*args, confirmation_seed=confirmation_seed, **kwargs)
        self._experiment_selection_seed = confirmation_seed + 104729
        self._experiment_rng = Random(self._experiment_selection_seed)

    def restore_practice_research(self, previous, previous_workspace, laboratory, *, confirmation_seed):
        restored = super().restore_practice_research(previous, previous_workspace, laboratory,
                                                     confirmation_seed=confirmation_seed)
        # Each saved random selection consumed one draw. Restore that same stream
        # without rerunning a comparison or purchasing its observations again.
        for comparison in self.experiment_comparisons:
            control = comparison.get('experiment_selection_control', {})
            if control.get('policy') == 'uniform_random_same_observation_cost':
                self._experiment_rng.choice(control['eligible_experiments'])
        return restored

    def _execute(self, action, arguments, laboratory):
        result = super()._execute(action, arguments, laboratory)
        if action != 'run' or arguments.get('operator') != 'compare_experiments':
            return result
        comparison = result['comparison']
        greedy = comparison.get('selected_experiment')
        if not greedy:
            return result

        # The common operator has already fit and forecast every proposed
        # experiment. Changing its selection buys no data and refits no model.
        record = self.experiment_comparisons[comparison['index']]
        choices = [experiment for experiment in record['comparisons']
                   if experiment['observation_cost'] == greedy['observation_cost']]
        chosen = self._experiment_rng.choice(choices)
        return _replace_selection(self, result, choices, chosen, {
            'policy': 'uniform_random_same_observation_cost',
            'seed': self._experiment_selection_seed})


class VarianceExperimentMethodAgent(ExperimentalDesignMethodAgent):
    """Select maximum prediction variance from the same-cost comparison pool."""

    def _execute(self, action, arguments, laboratory):
        result = super()._execute(action, arguments, laboratory)
        if action != 'run' or arguments.get('operator') != 'compare_experiments':
            return result
        comparison = result['comparison']
        original = comparison.get('selected_experiment')
        if not original:
            return result
        choices = [experiment for experiment in comparison['comparisons']
                   if experiment['observation_cost'] == original['observation_cost']]
        chosen = max(choices, key=lambda experiment: experiment['predicted_separation'])
        return _replace_selection(self, result, choices, chosen, {
            'policy': 'maximum_prediction_variance_same_observation_cost'})
