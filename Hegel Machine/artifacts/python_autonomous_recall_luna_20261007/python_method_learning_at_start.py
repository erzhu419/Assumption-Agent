"""Practice existing research programs using actual archived Python predictors."""

from dataclasses import replace
import json

from .code_agent import python_model_context_matches
from .temporal_method_learning import METHOD_PATH
from .temporal_paired_method import FrozenMethodAgent, ProcedureConsumerAgent
from .temporal_paired_practice import PAIRED_INTERFACE, PairedMethodLearner


SEED_PYTHON_METHOD = '''import json
def select_actions(spec, records, memory):
    state = json.load(open('research_state.json'))
    steps = state.get('method_results', [])
    prior = memory['research_roles']['predictor']
    if not steps:
        return {'action': 'evaluate', 'arguments': {'model': prior}}
    if steps[-1]['feedback'].get('fresh_confirmation', {}).get('passed') is False:
        return {'stop': 'prior_failed_confirmation; revise this research program'}
    return {'model': prior}
'''


PYTHON_PRACTICE_INTERFACE = PAIRED_INTERFACE.replace(
    'from empty observations and the SAME frozen prior knowledge in a fresh workspace;',
    'after the common paid prefix and with the SAME frozen prior knowledge in a fresh workspace;'
).replace(
    'No records exist at the first call.',
    'Three paid 24-row control trajectories exist at the first call. Their cost belongs to this trial.'
).replace(
    "memory['research_roles'] gives instrument and predictor file paths for the earlier\nknowledge.",
    "memory['research_roles']['predictor'] gives the earlier confirmed Python predictor.\n"
    'Its archive retains original training files and dependencies; no prior instrument is supplied.'
).replace(
    'The exact same source executes up to eight steps per trial, without another LLM\n'
    'choosing intermediate actions. After publication it will also run from empty\n'
    'observations with another previously learned calibration.',
    'The exact same source executes up to twelve steps per trial, without another LLM\n'
    'choosing intermediate actions. Confirmation costs 24 fresh rows. After publication\n'
    'it will run after the same paid prefix with another actually learned Python prior.'
).replace(
    'The host never selects a candidate or an experimental strategy for you.',
    'Within a practice, your program selects the predictive candidates and experiments.'
) + '''
Author finish {path:'shared/consolidated_method.py',lesson:'...'} explicitly requests
paid confirmation of every still-unconfirmed case for the current source, in the
listed case order. Each case executes your program and charges the same remaining
author account. Before these practices, finish identifies the most recently
supported earlier source, excluding sources whose latest attempt on any required
case failed. If your current source fails a case, finish compares that earlier
source on this case and then completes its own missing cases. Confirmed cases for
that exact source retain their paid support. All comparison practices use the same
remaining author account. Failure of a reuse or revision requirement likewise
compares the earlier source by completing its own missing cases and requirements.
Only a source passing every case and all existing reuse
and revision requirements can publish. An earlier source passing this comparison
is selected and published; otherwise your current source remains selected and both
outcomes are returned for further research. publication_practices records the
current draft; incumbent_comparison records the earlier draft and selection.
Already-confirmed cases for this exact source are skipped; editing the source
invalidates that coverage. Writing alone never starts a practice.
method_decision.json summarizes current source coverage, the latest publication
failure, and every saved source's latest paid support and unresolved failures.
Each entry gives its source path and an explicit restore action. The summary
does not select a source; restoration still requires your own action. Missing
cases remain unconfirmed, and an old source may have no usable support.
Your program may also return an ordinary action:
{'action':'evaluate', 'arguments':{'model': memory['research_roles']['predictor']}}
or a run/query/write/finish action under the shared tool contract. evaluate scores
already-paid observations and returns mse; it does not buy new observations or
return fresh_confirmation. The result enters method_results before the next
invocation. {'stop':'reason'} leaves the trial unresolved. Returning {'model':path},
or {'action':'finish','arguments':{'model':path}}, scores the candidate and buys
24 new confirmation rows. Only this submission returns fresh_confirmation;
passing requires mse <= 1e-10. Paid evaluation alone never confirms a model.
Each method_results entry stores the exact returned dictionary in proposal:
a model declaration uses proposal.model; an ordinary action uses
proposal.action and proposal.arguments. Read its actual outcome in feedback.
Runtime field names differ from compact author-facing summaries. Inside
select_actions, research_state.json method_results entries expose
feedback.fresh_confirmation.observation_indices: the actual paid response-row
indices of that confirmation. records[i].row_indices contains the corresponding
response-row indices for trajectory record i. refit.split.records,
refit.split.training_records and refit.split.validation_records contain trajectory
record indices into records, not response-row indices or positions in a subset.
Compact practice feedback may omit these arrays. Supplied counterexample ledgers
summarize them as row_interval:[first_row,last_row], with inclusive endpoints;
row_interval is a summary field, not a field in runtime fresh_confirmation.
The trial result.json retains the actual method_results and paid_observations.
Inspect the inherited research source for its existing behavior. Preserve working
branches and learn revisions from the paid practice outcomes.
Only the final self-contained source and lesson transfer. Use the trial's roles,
paid evidence and public tools; no reads of other workspaces or study artifacts.
'''


class PreparedPythonMethodAgent(FrozenMethodAgent):
    """Prepare an ordinary paid episode, then use the original frozen executor."""

    def __init__(self, provider, workspace, **kwargs):
        ProcedureConsumerAgent.__init__(self, provider, workspace, **kwargs)
        self.frozen_method_source = None

    def bind_method(self, source):
        self._save(METHOD_PATH, source)
        self.frozen_method_source = source


class PythonArchiveMethodLearner(PairedMethodLearner):
    """Keep the existing authoring/publication loop with the current paid prefix."""

    def __init__(self, provider, workspace, *, knowledge, cases, lab_factory, config, prepare_trial):
        super().__init__(provider, workspace, knowledge=knowledge, cases=cases,
                         lab_factory=lab_factory, config=config)
        self.prepare_trial = prepare_trial
        self.common_interface = PYTHON_PRACTICE_INTERFACE.replace('CASE_LABEL', repr(next(iter(cases))))

    def _prepare_program_state(self):
        """Put existing paid support beside the author's available source choices."""
        coverage = self._current_source_coverage()
        trials = {trial['directory']: trial for trial in self.practice_trials}
        candidates = []
        for entry in json.loads(self.files['method_candidates.json']):
            latest = self._latest_source_practices(trials[entry['trial']]['method_source'])
            supported = sorted(case for case, trial in latest.items()
                if trial['status'] == 'completed' and
                (trial.get('fresh_confirmation') or {}).get('passed') is True)
            candidates.append({key: entry[key] for key in ('trial', 'source_path', 'current')} | {
                'supported_cases': supported,
                'failed_cases': sorted(set(latest) - set(supported)),
                'missing_cases': sorted(set(self.cases) - set(supported)),
                'restore_action': {'action': 'run', 'arguments': {
                    'operator': 'restore_method', 'trial': entry['trial']}}})
        previous = json.loads(self.files.get('resume_context.json', '{}')).get('last_action')
        events = ([previous] if previous is not None else []) + self.events
        failure = None
        for event in reversed(events):
            feedback = event.get('feedback', {})
            if event['action'] != 'finish' or not feedback.get('error'):
                continue
            failed = next((practice['case'] for practice in feedback.get('publication_practices', [])
                if practice.get('status') != 'completed' or
                (practice.get('fresh_confirmation') or {}).get('passed') is not True), None)
            failure = {'case': failed, 'error': feedback['error'], 'gate': {
                key: feedback[key] for key in ('unconfirmed_cases', 'confirmed_prior_reuse_cases',
                    'confirmed_revision_cases', 'selective_revision_coverage', 'scoped_revision_coverage')
                if key in feedback}}
            break
        summary = {'current_source_coverage': coverage,
            'latest_publication_failure': failure, 'candidates': candidates}
        self.files['method_decision.json'] = json.dumps(summary, indent=2)
        (self.workspace / 'method_decision.json').write_text(self.files['method_decision.json'])

    def _execute(self, action, arguments, laboratory):
        if action != 'finish' or arguments.get('path') != METHOD_PATH:
            return super()._execute(action, arguments, laboratory)
        source = (self.workspace / METHOD_PATH).read_text()
        incumbent = self._incumbent_trial(source)
        practices, failed_case = self._complete_publication_practice()
        current_gate_error = None
        if failed_case is None:
            result = super()._execute(action, arguments, laboratory)
            if result.get('published'):
                return {**result, 'publication_practices': practices,
                        'current_source_coverage': self._current_source_coverage()}
            current_gate_error = result
        comparison = {'old_trial': incumbent['directory'] if incumbent else None,
            'current_failure_case': failed_case, 'practices': [], 'selected': False,
            'status': 'no_supported_incumbent'}
        if current_gate_error is not None:
            comparison['current_gate_error'] = current_gate_error
        if incumbent is not None:
            super()._execute('run', {'operator': 'restore_method',
                'trial': incumbent['directory']}, laboratory)
            older_practices, older_failure = self._complete_publication_practice(first_case=failed_case)
            comparison['practices'] = older_practices
            comparison['status'] = 'incumbent_practice_failed'
            if older_failure is None:
                rejected = (f'failed on {failed_case}' if failed_case is not None
                            else 'failed publication requirements')
                result = super()._execute(action, {**arguments,
                    'lesson': f'Retained prior supported method after revision {rejected}; '
                              'confirmed its own required cases.'}, laboratory)
                comparison['publication_feedback'] = result
                if result.get('published'):
                    comparison.update(selected=True, status='incumbent_published')
                    return {**result, 'publication_practices': practices,
                        'incumbent_comparison': comparison,
                        'current_source_coverage': self._current_source_coverage()}
                comparison['status'] = 'incumbent_publication_requirements_unmet'
            super()._execute('write', {'path': METHOD_PATH, 'code': source}, laboratory)
        coverage = self._current_source_coverage()
        return {**(current_gate_error or {'error': 'method publication practice failed'}),
            'publication_practices': practices,
            'incumbent_comparison': comparison, 'current_source_coverage': coverage,
            'unconfirmed_cases': coverage['missing_cases']}

    def _incumbent_trial(self, current_source):
        """Choose the latest supported old source without an unresolved case failure."""
        seen = {current_source}
        for trial in reversed(self.practice_trials):
            source = trial['method_source']
            if source in seen:
                continue
            seen.add(source)
            outcomes = self._latest_source_practices(source)
            if outcomes and all(item['status'] == 'completed' and
                    (item.get('fresh_confirmation') or {}).get('passed') is True
                    for item in outcomes.values()):
                return trial
        return None

    def _complete_publication_practice(self, *, first_case=None):
        """Spend the ordinary remaining account on this source's missing support."""
        missing = set(self._current_source_coverage()['missing_cases'])
        practices = []
        order = ([first_case] if first_case in missing else [])
        order.extend(case for case in self.cases if case in missing and case != first_case)
        for case in order:
            outcome = self._practice(case)
            practices.append({'case': case, **outcome})
            if (outcome.get('status') != 'completed' or
                    (outcome.get('fresh_confirmation') or {}).get('passed') is not True):
                return practices, case
        return practices, None

    def _practice(self, case):
        if case not in self.cases:
            raise ValueError('choose ' + ' or '.join(self.cases))
        remaining = self.config.observation_budget - self.costs['observation_cost']
        compute = self.config.compute_budget - self.costs['compute_cost']
        seconds = self.config.execution_seconds - self.costs['python_execution_seconds']
        if remaining < 96 or compute < 96 or seconds <= 0:
            return {'error': 'remaining resources cannot cover the paid prefix and fresh confirmation'}
        limits = replace(self.config, observation_budget=min(384, remaining), confirmation_rows=24,
            compute_budget=min(360000, compute), execution_seconds=min(120., seconds),
            max_model_calls=0, token_budget=0)
        source = (self.workspace / METHOD_PATH).read_text()
        directory = f'trials/trial_{len(self.practice_trials):03d}'
        payload = self.prepare_trial(self.cases[case], self.workspace / directory, source, limits)
        agent = payload['agent']
        # Match paired practice: each attempt gets a new confirmation stream,
        # including after restoration of earlier paid trials.
        agent.confirmation_rng.seed(self.cases[case]['confirmation_seed'] + len(self.practice_trials))
        result = agent.execute_method(payload['laboratory'], max_steps=12)
        selected_prior = bool(result.get('model') and any(
            python_model_context_matches(agent.workspace, result['model'],
                candidate['snapshot_workspace'], candidate['inner_model'])
            for candidate in payload['inherited_candidates']))
        return self._record_practice(case, directory, source, agent, result, selected_prior)
