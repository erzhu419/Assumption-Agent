"""Practice existing research programs using actual archived Python predictors."""

import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from .code_agent import python_model_context_matches
from .checkpoint_storage import write_text_incrementally
from .online_inquiry_llm import InquiryLLMConfig
from .temporal_method_learning import METHOD_PATH
from .temporal_paired_method import FrozenMethodAgent, ProcedureConsumerAgent
from .temporal_paired_practice import (
    PAIRED_INTERFACE, PairedMethodLearner, candidate_execution_summary, practice_execution_summary,
)


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
Its latest_practice and latest_evaluated_practice retain compact execution_summary
facts: actual research returns, candidate paths, evaluations and submissions.
These two views are chronological and may describe an earlier comparison source.
current_source_latest_failure and current_source_latest_evaluation identify the
selected program's own evidence; latest_incumbent_comparison keeps the old
source's outcome separate. Each practice view states source_matches_current.
Full trial result.json uses method_results; author practice feedback uses steps.
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

    def __init__(self, provider, workspace, *, knowledge, cases, lab_factory, config, prepare_trial,
                 practice_limits=None, paid_trace_references=None):
        super().__init__(provider, workspace, knowledge=knowledge, cases=cases,
                         lab_factory=lab_factory, config=config, paid_trace_references=paid_trace_references)
        self.prepare_trial = prepare_trial
        self._paid_execution_summaries = {}
        self.practice_limits = practice_limits or InquiryLLMConfig(
            observation_budget=384, confirmation_rows=24, compute_budget=360000,
            execution_seconds=120., max_model_calls=0, token_budget=0)
        self.common_interface = PYTHON_PRACTICE_INTERFACE.replace('CASE_LABEL', repr(next(iter(cases))))
        self.common_interface = self.common_interface.replace(
            'Confirmation costs 24 fresh rows.',
            f'Confirmation costs {self.practice_limits.confirmation_rows} fresh rows.').replace(
            '24 new confirmation rows.', f'{self.practice_limits.confirmation_rows} new confirmation rows.')

    def result(self, status):
        # A final edit can complete a practice on the last allowed model call.
        # Refresh the same decision view for checkpoints and terminal exports,
        # even when no subsequent request will prepare it again.
        self._prepare_program_state()
        return super().result(status)

    def _prepare_program_state(self):
        """Put existing paid support beside the author's available source choices."""
        coverage = self._current_source_coverage()
        trials = {trial['directory']: trial for trial in self.practice_trials}
        entries = json.loads(self.files['method_candidates.json'])
        if self.practice_trials:
            latest_trial = self.practice_trials[-1]
            latest_summary = candidate_execution_summary(
                self._paid_practice_view(latest_trial)['execution_summary'])
            for entry in entries:
                for evidence in entry['trials']:
                    if evidence['directory'] == latest_trial['directory']:
                        evidence['execution_summary'] = latest_summary
            # The inline view and the advertised file must describe the same
            # actual execution. Retain the historical trial ledger unchanged.
            content = json.dumps(entries, indent=2)
            write_text_incrementally(self.workspace / 'method_candidates.json', content)
            self.files['method_candidates.json'] = content
        candidates = []
        for entry in entries:
            # Historical program selection must retain paid outcomes from earlier
            # assignments. Current-work confirmation is a separate question and
            # continues to use the learner's stricter coverage/finish methods.
            latest = PairedMethodLearner._latest_source_practices(
                self, trials[entry['trial']]['method_source'])
            supported = sorted(case for case, trial in latest.items()
                if trial['status'] == 'completed' and
                (trial.get('fresh_confirmation') or {}).get('passed') is True)
            candidates.append({key: entry[key] for key in ('trial', 'source_path', 'current')} | {
                'evidence_scope': 'Historical executions of this exact source, across research '
                    'assignments; these guide program selection, not current-work confirmation.',
                'supported_cases': supported,
                'failed_cases': sorted(set(latest) - set(supported)),
                'missing_cases': sorted(set(self.cases) - set(supported)),
                'latest_case_evidence': [{
                    **{key: deepcopy(trial[key]) for key in
                       ('case', 'directory', 'status', 'fresh_confirmation', 'costs',
                        'prior_research_cost', 'continuation_cost') if key in trial},
                    'result_file': str(Path(trial['directory']) / 'result.json'),
                    **({'execution_summary': candidate_execution_summary(
                        self._paid_practice_view(trial)['execution_summary']
                        if trial is self.practice_trials[-1]
                        else trial['execution_summary'])}
                       if trial.get('execution_summary') else {})}
                    for _, trial in sorted(latest.items())],
                'restore_action': {'action': 'run', 'arguments': {
                    'operator': 'restore_method', 'trial': entry['trial']}}})
        previous = json.loads(self.files.get('resume_context.json', '{}')).get('last_action')
        events = ([previous] if previous is not None else []) + self.events
        failure, incumbent_comparison = None, None
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
            comparison = feedback.get('incumbent_comparison')
            if comparison is not None:
                incumbent_comparison = {key: comparison[key] for key in
                    ('old_trial', 'current_failure_case', 'selected', 'status') if key in comparison}
                incumbent_comparison['practices'] = [self._paid_practice_view(trials[item['directory']])
                    for item in comparison.get('practices', []) if item.get('directory') in trials]
            break
        # The paid-practice ledger already survives resumption and is updated
        # after each new practice. Surface its outcome directly: replacing the
        # large resume_context file with a file reference otherwise hid the
        # interruption that the author was being asked to respond to.
        latest_practice = (self._paid_practice_view(self.practice_trials[-1])
                           if self.practice_trials else None)
        current_source = (self.workspace / METHOD_PATH).read_text()
        try:
            definitions = [{'line': node.lineno, 'end_line': node.end_lineno}
                for node in ast.parse(current_source).body
                if isinstance(node, ast.FunctionDef) and node.name == 'select_actions']
            source_entry = {'path': METHOD_PATH, 'name': 'select_actions',
                'effective_definition': definitions[-1] if definitions else None,
                'shadowed_definitions': definitions[:-1]}
        except SyntaxError as error:
            source_entry = {'path': METHOD_PATH, 'syntax_error': str(error)}
        current_latest = self._latest_source_practices(current_source)
        current_failure = next((trial for trial in reversed(self.practice_trials)
            if current_latest.get(trial['case']) is trial and not
            (trial['status'] == 'completed' and
             (trial.get('fresh_confirmation') or {}).get('passed') is True)), None)
        current_evaluation = next((trial for trial in reversed(self.practice_trials)
            if trial['method_source'] == current_source and
            trial.get('execution_summary', {}).get('latest_evaluation')), None)
        stopped = next((trial for trial in reversed(self.practice_trials)
            if trial['method_source'] == current_source and trial.get('no_progress')), None)
        latest_no_progress = self._paid_practice_view(stopped) if stopped is not None else None
        evaluated = next((trial for trial in reversed(self.practice_trials)
            if trial.get('execution_summary', {}).get('latest_evaluation')), None)
        latest_evaluated = self._paid_practice_view(evaluated) if evaluated is not None else None
        summary = {'current_source_coverage': coverage, 'current_source_entry': source_entry,
            'current_source_latest_failure': (self._paid_practice_view(current_failure)
                                              if current_failure is not None else None),
            'current_source_latest_evaluation': (self._paid_practice_view(current_evaluation)
                                                 if current_evaluation is not None else None),
            'latest_publication_failure': failure, 'candidates': candidates,
            'latest_incumbent_comparison': incumbent_comparison,
            'latest_practice': latest_practice,
            'latest_evaluated_practice': latest_evaluated,
            'current_source_latest_no_progress': latest_no_progress}
        comparison = self._failed_case_paid_comparison()
        if comparison is not None:
            summary['failed_case_paid_comparison'] = comparison
        content = json.dumps(summary, indent=2)
        write_text_incrementally(self.workspace / 'method_decision.json', content)
        self.files['method_decision.json'] = content

    def _paid_practice_view(self, trial):
        directory = trial['directory']
        summary = trial.get('execution_summary', {})
        submitted = summary.get('latest_submitted_predictor', {})
        native = summary.get('latest_native_result')
        # Refresh the most recent paid practice when its saved short view
        # predates the native decision feedback. Older large traces remain
        # available by their recorded paths, without a history-wide reread.
        missing_native_decision = (self.practice_trials and trial is self.practice_trials[-1]
            and native is not None and 'research_decision' not in native)
        missing_submitted_source = (self.practice_trials and trial is self.practice_trials[-1]
            and submitted.get('model_file') and 'source_preview' not in submitted)
        missing_execution_cost = (self.practice_trials and trial is self.practice_trials[-1]
            and 'continuation_cost' in trial and 'practice_execution_cost' not in summary)
        missing_author_work = (self.practice_trials and trial is self.practice_trials[-1]
            and trial.get('status') == 'author_work_request' and 'latest_author_work_request' not in summary)
        if ('research_steps' not in summary or
                (summary.get('submission_count', 0) and 'latest_submitted_predictor' not in summary) or
                ('paid_rows' in submitted and 'submission_evidence' not in submitted) or
                missing_native_decision or missing_submitted_source or missing_execution_cost or missing_author_work):
            if directory not in self._paid_execution_summaries:
                path = self.workspace / directory / 'result.json'
                self._paid_execution_summaries[directory] = (
                    practice_execution_summary(json.loads(path.read_text()), directory)
                    if path.is_file() else summary)
            summary = self._paid_execution_summaries[directory]
        return {key: trial[key] for key in
            ('case', 'directory', 'status', 'costs', 'fresh_confirmation', 'no_progress',
             'prior_research_cost', 'continuation_cost', 'resume_origin', 'practice_start') if key in trial} | {
            'source_path': f'{directory}/{METHOD_PATH}',
            'source_matches_current': trial['method_source'] == (self.workspace / METHOD_PATH).read_text(),
            'execution_summary': summary}

    def _failed_case_paid_comparison(self):
        """Return an existing same-case success, without buying or selecting it."""
        if not self.practice_trials:
            return None
        failed = self.practice_trials[-1]
        if (failed['status'] == 'completed' and
                (failed.get('fresh_confirmation') or {}).get('passed') is True):
            return None
        seen = {failed['method_source']}
        for trial in reversed(self.practice_trials[:-1]):
            source = trial['method_source']
            if trial['case'] != failed['case'] or source in seen:
                continue
            # Only the latest outcome of this source on this case supplies
            # support. Its failures on other cases do not erase this evidence.
            seen.add(source)
            if (trial['status'] == 'completed' and
                    (trial.get('fresh_confirmation') or {}).get('passed') is True):
                return {'failed_practice': self._paid_practice_view(failed),
                    'previous_supported_practice': self._paid_practice_view(trial)}
        return None

    def _record_practice(self, case, directory, source, agent, result, selected_prior):
        feedback = super()._record_practice(case, directory, source, agent, result, selected_prior)
        comparison = self._failed_case_paid_comparison()
        if comparison is not None:
            feedback['failed_case_paid_comparison'] = comparison
        return feedback

    def _execute(self, action, arguments, laboratory):
        if action != 'finish' or arguments.get('path') != METHOD_PATH:
            return super()._execute(action, arguments, laboratory)
        source = (self.workspace / METHOD_PATH).read_text()
        incumbent = self._incumbent_trial(source)
        practices, failed_case = self._complete_publication_practice()
        source = (self.workspace / METHOD_PATH).read_text()
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
            if outcome.get('repaired_candidate_confirmation', {}).get('retained'):
                # Practice selected and independently confirmed a repaired draft.
                # Complete that draft's coverage, not the pre-repair case list.
                remaining, failed = self._complete_publication_practice()
                return practices + remaining, failed
            if (outcome.get('status') != 'completed' or
                    (outcome.get('fresh_confirmation') or {}).get('passed') is not True):
                return practices, case
        return practices, None

    def _prepare_practice(self, agent, laboratory):
        pass

    def _practice(self, case):
        if case not in self.cases:
            raise ValueError('choose ' + ' or '.join(self.cases))
        remaining = self.config.observation_budget - self.costs['observation_cost']
        compute = self.config.compute_budget - self.costs['compute_cost']
        seconds = self.config.execution_seconds - self.costs['python_execution_seconds']
        limit = self.practice_limits
        observations = min(limit.observation_budget, remaining)
        compute = min(limit.compute_budget, compute)
        seconds = min(limit.execution_seconds, seconds)
        initial_observations = getattr(self.prepare_trial, 'initial_observation_cost', 72)
        required = initial_observations + limit.confirmation_rows
        if observations < required or compute < required or seconds <= 0:
            prefix = 'the paid prefix and ' if initial_observations else ''
            return {'error': f'remaining resources cannot cover {prefix}fresh confirmation'}
        limits = replace(limit, observation_budget=observations,
            compute_budget=compute, execution_seconds=seconds,
            max_model_calls=0, token_budget=0)
        source = (self.workspace / METHOD_PATH).read_text()
        directory = f'trials/trial_{len(self.practice_trials):03d}'
        payload = self.prepare_trial(self.cases[case], self.workspace / directory, source, limits)
        agent = payload['agent']
        self._prepare_practice(agent, payload['laboratory'])
        prior_cost = payload.get('prior_research_cost')
        if prior_cost is None:
            # Fresh practices have separate streams. A restored live scene has
            # already advanced its original stream past the retained paid tests.
            agent.confirmation_rng.seed(self.cases[case]['confirmation_seed'] + len(self.practice_trials))
        result = agent.execute_method(payload['laboratory'], max_steps=12)
        if prior_cost is not None:
            result['prior_research_cost'] = deepcopy(prior_cost)
            result['continuation_cost'] = {key: value - prior_cost.get(key, 0)
                if value is not None else None for key, value in result['costs'].items()}
            result['practice_start'] = {
                'method_results': payload['prior_method_count'],
                'events': payload['prior_event_count']}
            if payload.get('resume_origin'):
                result['resume_origin'] = payload['resume_origin']
        selected_prior = bool(result.get('model') and any(
            python_model_context_matches(agent.workspace, result['model'],
                candidate['snapshot_workspace'], candidate['inner_model'])
            for candidate in payload['inherited_candidates']))
        return self._record_practice(case, directory, source, agent, result, selected_prior)
