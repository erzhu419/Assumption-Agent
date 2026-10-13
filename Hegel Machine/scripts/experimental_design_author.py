"""Study-specific feedback for learning prospective experimental design.

Predictive practice support remains intact when the narrower research objective
is unresolved. This helper does not generate hypotheses or laboratory actions.
"""

from copy import deepcopy
import json

from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH


FEEDBACK_FILE = 'experiment_design_feedback.json'


def design_practice_feedback(result):
    # Imported here because the existing report module also imports the study
    # runner. Keep its scientific trace interpretation in one place.
    from scripts.run_anonymous_online_research import experimental_design_trace

    trace = experimental_design_trace(result)
    revisions = []
    for revision in trace['revisions']:
        missing = []
        readout_recorded = bool(revision['readout_status'] in {'supported', 'refuted', 'inconclusive'}
            and not revision['readout_error'] and revision['analysis_position'] is not None
            and set(revision['readout_evidence']).intersection(revision['paid_row_indices']))
        requirements = (
            (len(revision['supported_models']) >= 2,
             'Competing runnable predictions supported by the existing paid evidence.'),
            (revision['predictions_differ'], 'Different predictions before acquisition.'),
            (revision['selected_from_recorded_comparison'],
             'The acquired experiment selected from the recorded prospective comparison.'),
            (bool(revision['paid_row_indices']), 'Actual new observations from the experiment.'),
            (readout_recorded, 'An authored readout citing the newly acquired rows.'),
            (revision['before_first_submission'] and revision['readout_before_first_submission'],
             'Acquisition and readout before the first final model submission.'),
            (revision['measured_outcome'] in {'supported_and_refuted', 'hypothesis_set_refuted'},
             'A measured rejection or revision of the competing explanations.'),
            (revision['followed_by_accepted_model'], 'A subsequently confirmed final explanation.'),
        )
        missing.extend(description for holds, description in requirements if not holds)
        submission = trace['first_submission']
        subsequent_submission = (deepcopy(submission) if submission is not None
            and revision['analysis_position'] is not None
            and revision['analysis_position'] < submission['position'] else None)
        revisions.append({key: deepcopy(revision[key]) for key in (
            'index', 'supported_models', 'predictions_differ', 'measured_outcome',
            'measured_supported_models', 'measured_refuted_models', 'authored_readout',
            'before_first_submission', 'readout_before_first_submission',
            'selected_from_recorded_comparison', 'followed_by_accepted_model')}
            | {'readout_recorded': readout_recorded, 'subsequent_submission': subsequent_submission,
               'design_supported': not missing, 'missing': missing})
    prediction_supported = bool(result.get('status') == 'completed' and result.get('model')
        and (result.get('fresh_confirmation') or {}).get('passed') is True)
    supported = prediction_supported and any(item['design_supported'] for item in revisions)
    return {'prediction_supported': prediction_supported, 'design_supported': supported,
        'final_confirmation': trace['final_confirmation'], 'revisions': revisions,
        'explicit_acquired_rows': sum(item['paid_rows'] for item in trace['explicit_queries']),
        'missing': ([] if supported else
            ['No current successful practice records prospective rival-driven selection, '
             'its measured consequence and an authored readout before final submission.']),
        'scope': 'readout_recorded identifies an executed authored analysis citing acquired evidence. '
            'subsequent_submission records the first submission when it follows that analysis. '
            'Neither establishes that the verdict changed the decision or that a representation '
            'family was refuted or revised. Practice support does not establish independent '
            'transfer or a causal benefit of experiment selection.'}


class ExperimentalDesignMethodLearner(AnonymousResearchMethodLearner):
    """Continue paid authoring after predictive publication until its design task is met."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictive_publications = []

    def _practice(self, case):
        feedback = super()._practice(case)
        if 'directory' in feedback:
            trial = next(item for item in reversed(self.practice_trials)
                         if item['directory'] == feedback['directory'])
            result = json.loads((self.workspace / trial['directory'] / 'result.json').read_text())
            trial['experimental_design'] = design_practice_feedback(result)
            feedback['experimental_design'] = deepcopy(trial['experimental_design'])
        self._design_feedback()
        return feedback

    def _design_feedback(self):
        source = (self.workspace / METHOD_PATH).read_text()
        latest = self._latest_source_practices(source)
        practices = []
        for case, trial in latest.items():
            if 'experimental_design' not in trial:
                result = json.loads((self.workspace / trial['directory'] / 'result.json').read_text())
                trial['experimental_design'] = design_practice_feedback(result)
            # Restored paid practices retain their cached report. Rename its
            # factual label without re-reading or re-executing that experiment.
            for revision in trial['experimental_design'].get('revisions', []):
                if 'readout_consumed' in revision:
                    revision['readout_recorded'] = revision.pop('readout_consumed')
            practices.append({'case': case, 'directory': trial['directory'],
                              **deepcopy(trial['experimental_design'])})
        supported = [trial['case'] for trial in practices if trial['design_supported']]
        report = {'design_supported_cases': supported, 'practices': practices,
            'predictive_publications': deepcopy(self.predictive_publications),
            'feedback_scope': 'Only latest practices on the current exact source and listed cases count. '
                'Prediction support and the earlier method remain available. At least one actual '
                'prospective design chain is needed across the practices; other cases may directly '
                'reuse an already supported explanation. No tool-call count is required. '
                'A recorded readout establishes evidence analysis; whether its verdict changed '
                'a later decision remains a separate question.'}
        self.files[FEEDBACK_FILE] = json.dumps(report, indent=2)
        (self.workspace / FEEDBACK_FILE).write_text(self.files[FEEDBACK_FILE])
        return report

    def _execute(self, action, arguments, laboratory):
        feedback = super()._execute(action, arguments, laboratory)
        design = self._design_feedback()
        if action == 'finish' and feedback.get('published'):
            source = self.published_source
            latest = self._latest_source_practices(source)
            publication = {'path': self.model, 'confirmed_cases': sorted(latest),
                'source_trial': next(reversed(latest.values()))['directory'], 'lesson': self.lesson}
            if publication not in self.predictive_publications:
                self.predictive_publications.append(publication)
            design = self._design_feedback()
            if not design['design_supported_cases']:
                self.model, self.published_source = None, None
                feedback.pop('published')
                feedback.update(task_status='prediction_supported_design_unresolved',
                    predictive_publication=publication,
                    missing='Prediction passed all required practices. The current program has no '
                        'recorded prospective rival-driven experiment followed by measured hypothesis '
                        'revision, authored readout and final confirmation. Continue revising the '
                        'research method within the same remaining account; all paid support is retained.')
            else:
                feedback['task_status'] = 'prediction_and_design_supported'
        feedback['experimental_design'] = design
        return feedback

    def result(self, status):
        result = super().result(status)
        design = self._design_feedback()
        result['predictive_publications'] = deepcopy(self.predictive_publications)
        result['predictive_publication'] = (deepcopy(self.predictive_publications[-1])
                                            if self.predictive_publications else None)
        supported = bool(design['design_supported_cases'])
        result['design_objective'] = {'supported': supported,
            'supported_cases': design['design_supported_cases'], 'feedback_file': FEEDBACK_FILE,
            'task_status': ('prediction_and_design_supported' if self.published_source and supported else
                'prediction_supported_design_unresolved' if result['predictive_publication'] else
                'researching')}
        return result
