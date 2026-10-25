"""Continue authored research with a common archive and live question records.

Ordinary Code and Hegel share execution, evidence, notebooks and revision tools.
Hegel additionally keeps its public research root and executable source manifest.
"""

from copy import deepcopy
import json
from math import isfinite
import os
from pathlib import Path
from random import Random
import re
import shutil

from .online_inquiry_llm import COMMON_INTERFACE, open_assigned_question
from .checkpoint_storage import (
    write_text_incrementally, retain_checkpoint_sources, read_json_checkpoint,
)
from .question_continuation import QUESTION_INTERFACE, step_question, pending_query_arguments
from .question_routing import ROUTING_INTERFACE, route_research, continue_conditional_plan
from .theory_consolidation import (
    CONSOLIDATION_INTERFACE, LIBRARY_PATH, consolidate_research, transfer_theory,
    INDEX_PATH, install_theory_library, install_theory_index, save_theory_index,
    _save as save_theory_library,
)
from .inquiry_session import begin_session, enter_notebook, leave_notebook
from .representation_inquiry import open_representation_inquiry, record_representation_return
from .scientific_generalization import (
    generalization_failures, prepare_generalization_review, PREDICTION_METHOD_QUESTION,
)
from .structural_inquiry import (
    STRUCTURAL_RESEARCH_INTERFACE, execute_structural_research, structural_summary,
    NATIVE_CONDITION_INTERFACE, execute_native_condition_research, native_condition_method,
    native_evidence_extension_review,
)
from .structural_proposal import (
    open_structural_proposal, trial_structural_proposal, record_structural_return,
    structural_proposal_context, extract_existing_return_evaluation, STRUCTURAL_PROPOSAL_INSTRUCTION,
)
from .evidence_readout_proposal import (
    prepare_evidence_readout, trial_evidence_readout, evidence_readout_context,
    EVIDENCE_READOUT_INSTRUCTION,
)
from .anonymous_history_lab import AnonymousHistoryLab
from .code_agent import (
    _archived_paid_trace, _bound_model_contexts, _copy_execution_context, _copy_model_file,
    _model_context_ignored, python_model_context, literal_python_model_context_matches,
)
from .paid_score_history import (
    restore_score_history_value, retain_event_history, restore_comparison_history,
)
from .code_provider_context import _saved_context_reference, load_code_context
from .temporal_operator_agent import OPERATOR_INTERFACE
from .inquiry_revision import (
    analyze_revision, analyze_paid_evidence, compare_experiments, revision_evidence_scope,
)
from .executable_evidence import (
    EVIDENCE_INTERFACE, before_candidate, bind_components, materialize, record_answer,
    record_evaluation, research_repairs, _local_relations,
)
from .observable_research import (
    OBSERVABLE_INTERFACE, dependency_focus, evaluate_observable, refresh_observables,
)
from .conditional_research import (
    CONDITIONAL_INTERFACE, evaluate_component, refresh_components,
)
from .temporal_paired_method import (
    FROZEN_METHOD_INTERFACE,
    PROCEDURE_INTERFACE,
    ProcedureConsumerAgent,
)


_HOST_RECORD_FILES = {
    "active.json", "training.json", "archive.json", "evaluation.json",
    "result.json", "protocol.json", "summary.json",
}

_SOURCE_EXECUTION_REQUIREMENT = (
    "During source learning, Hegel must actually query at least once through an authored\n"
    "select_actions program under an explicit current question; save that program.\n"
)

_CONTINUATION_REQUIREMENT = """This task continues an existing executable research archive.
The Hegel manifest may retain its existing research programs. A new program or
query is optional; ordinary evaluation and direct finish are available.
For a consumer task, finish registers your explicitly selected model path in the
current manifest if missing. Existing research entries are preserved. Registration
adds no scientific support: fresh predictive confirmation still determines adoption.
"""


# Reuse the existing mechanism-independent whole-trajectory refit contract.
_REFIT_INTERFACE = OPERATOR_INTERFACE[OPERATOR_INTERFACE.index("run {operator:'refit'"):
    OPERATOR_INTERFACE.index("run {operator:'legacy_capabilities'")]


COMMON_ARCHIVE_INTERFACE = COMMON_INTERFACE.replace(
    "Textual memory receives only earlier agent-authored lessons, not earlier code.\n"
    "Code/ReAct receives no previous-task knowledge. Task-local tools remain equal.\n",
    "Both Code/ReAct and Hegel receive the same supplied archive: executable files,\n"
    "lessons, the manifest and earlier question traces. Read, execute, revise or\n"
    "decline its contents using the same ordinary Python tools. Archive entries\n"
    "can include unsuccessful investigations; their presence is not confirmation.\n"
    "Both conditions retain actual authored code, including unfinished source\n"
    "work. Code/ReAct may organize this archive freely and has no requirement to\n"
    "create a manifest or question hierarchy. Hegel retains an executable manifest\n"
    "under the current research phase's requirements.\n",
) + FROZEN_METHOD_INTERFACE + PROCEDURE_INTERFACE + _REFIT_INTERFACE


ONLINE_RESEARCH_INTERFACE = """
This is a continuing online investigation. Each ordinary action returns to the
same live researcher with its real execution feedback. research_notes is your
own persistent notebook. online_research identifies existing questions, paid
evidence awaiting analysis, and child answers not yet reviewed by their parent.
These are research records, not a host-selected scientific next step.
Use stable question IDs. A question's assignment retains its original purpose;
its question text may become a narrower hypothesis as the investigation proceeds.
Optional research_question on any action associates that decision with a question.
An association records your attention, not scientific support or a changed model.
The actual experiment question may be supplied in query.question as before.
No child or additional observation is required when the paid evidence suffices.

run {operator:'inquire',question:{id:'local',question:'your inquiry',parent_id:'task',depends_on:[]},
max_model_calls:8} delegates your chosen question to a continuing local researcher.
It uses this same model, workspace, paid observations, apparatus and total account.
Each question retains its own notebook and actual feedback; the same ID resumes it.
The original assignment remains separate from its evolving scientific hypothesis.
The child can write/run/repair code, query, evaluate, analyze evidence, bind models,
or explicitly delegate another question. You choose when delegation is useful.
A supported or refuted answer to that assignment returns to the caller. Otherwise
finish {lesson:'...',artifacts:['shared/child.py']} returns existing authored files
and the unresolved finding to its parent; local finish never submits the response
predictor or grants scientific support. The parent chooses whether to use the result.
Only root finish requests fresh predictive confirmation. Local calls count against
every enclosing inquiry and the overall account, retaining one decision for the
caller to use or revise the returned work. Frozen zero-LLM methods cannot
invoke this live researcher; their research must execute as ordinary Python.

evaluate {model:'shared/candidate.py',code:'ordinary predictor Python source'}
saves exactly your supplied source and immediately executes ordinary paid-data
evaluation and old-candidate retention. Omitting code evaluates the existing file.
This does not submit a model or buy observations. A later finish selects the model
for fresh confirmation. Use actual evaluation and confirmation feedback to decide
whether to revise its source, construct an experiment, or stop.

run {operator:'refit',model:'shared/hypothesis.py',code:'your fit/predict Python source',
path:'shared/hypothesis_fit'} likewise saves your supplied program, then runs the
existing separate-trajectory refit validation. The returned path is the isolated
candidate for evaluate, compare_experiments or finish; validation is not fresh
confirmation. Omitting code refits the existing file without changing its source.

The research deliverable is an executable predictor you choose and its actual
evaluation. A diagnostic script or printed validation score alone does not install
a predictor. When an analysis warrants a candidate, save its fit/predict program
using the write or evaluate action, then use its executed feedback to revise or
submit it for independent confirmation. The existing evaluation path retains the
better supported prior draft. If the evidence supports no candidate, retain that
unresolved finding. You choose which analyses and representations to develop.

run {operator:'analyze_revision',revision_index:0,path:'shared/readout.py'}
executes your readout(result,public_spec,memory) on an already purchased revision.
Use it to complete or repair analysis without repeating acquisition. The saved
pre-experiment predictions, sources, paid cutoff, and observations remain fixed;
each executed readout is retained. The resulting answer returns to that revision's
question and parent. You choose whether and how it changes the next research step.
online_research.recent_revisions gives the actual record indices, row ranges and
forecast scope. Re-reading an old revision does not analyze newer experiments.
For those, use run {operator:'analyze_evidence',record_indices:[...],path:'shared/readout.py'}
with actual indices from online_research.recent_paid_records; this has no old forecasts.
Full questions, revision records and procedure feedback remain in research_state.json.
Each question's executed_analyses indexes its actual ordinary-Python runs, including
errors and unfinished work. These execution records do not establish an answer.
The active question and its ancestors retain their analysis outputs in evidence_notebook.

method_decision.methods keeps each registered program's execution action and
separate prior practice record. After a program's failed confirmation, continuation
identifies the submitted predictor, the new paid evidence and an action to invoke
that program again. It will receive the expanded records. You decide whether to
continue, edit revision_path, change the research route or stop. The failed test
evaluated that predictor; the program's ability to revise it remains a separate
question. A confirmation execution error is not a measured prediction mismatch.
submitted_candidate names the saved predictor and fitting view from before that
confirmation, even if a later method invocation overwrites its original file.
submitted_candidates keeps these attempts after a method stops. A method's stop
reason describes its own attempt; it does not exclude every model in its family.
method_decision.research_options exposes retained research operators for the current
question, their finite priors, callable actions and actual current-task results.
Choose a method, revise its inputs, or continue ordinary Python from the evidence.

run {operator:'compare_experiments',target:'response',models:['shared/a.py','shared/b.py'],
experiments:[{id:'an_authored_experiment',actions:[public_action,...]},...]}
executes your candidate predictors on your proposed public actions without buying
observations. It records paid fit and prospective forecasts for every offered
predictor; automatic experiment selection uses only candidates meeting the paid
MSE criterion. Failed or refuted candidates remain in the report. When none meets
that criterion, the forecasts can still inform your own diagnostic experiment;
ordinary query with revision.models freezes your chosen provisional predictors
before purchase. Actual evidence may reject all of them.
The returned comparison includes each authored experiment's forecast, and may
suggest the one with greatest mean prediction variance. A suggestion is neither
an acquisition nor evidence that its scientific alternatives are exhaustive.
query_arguments contains ordinary actions and revision arguments you can choose
to use with query; you may add your readout and question, change the experiment,
or proceed to confirmation. The suggested query names saved candidate contexts;
query reexecutes those contexts before purchase and charges that new scoring.
Their fitting views stay fixed. A refitted candidate is a new explanation to
evaluate and compare on current paid evidence.
To value an experiment for a prediction task, supply reference_actions:[public_action,...]
in the same comparison. These describe the control histories on which future
predictions matter. They acquire no observations and expose no reference labels.
The existing finite-model risk calculation then ranks expected reference-MSE
reduction per observation, using the same supported candidates and fitting views.
task_value reports the reference risk and each option's remaining expected risk;
equal or zero values are valid. This is conditional on your candidate set and
reference inputs, not proof that a model family is true or exhausted. Offer
different supported parameter fits when the existing evidence cannot identify
one fit; a single fitted instance does not represent every member of its family.
Omit target for the response, or name a public channel such as 'meter'. A channel
comparison uses only purchased readings of that target for paid fit and freezes
them in each new candidate context. Already retained candidates keep their own
fitting views. Every proposed action must buy that channel through channels.
The suggested revision keeps the target and scores its acquired readings; these
auxiliary candidates remain separate from the retained response predictor.
Full sources and predictions persist in research_state.json experiment_comparisons;
online_research.latest_comparison summarizes the most recent comparison.
"""


def record_revision_comparison_failures(agent, question, comparison, *, source_ref,
                                      comparison_question=None):
    """Consume complete paid failures of exact exported revision instances.

    The ordinary comparison has already executed. Its retained assessment owns
    the actual Python context, ordered paid rows and residuals. No Program is
    predicted or fitted here, and rejected drafts do not become adopted source
    revisions. Existing incomplete point checks remain incomplete.
    """
    from dataclasses import asdict
    from .language import Program
    from .research_episodes import SEMANTIC_FIELDS, _child_ref, load_source_values, resolve_program_sources
    from .temporal_operator_agent import export_temporal_program
    from .theory_consolidation import (
        _absolute_reference, _anchor_file_references, _reference_workspace,
        library_entry_reference,
    )

    result = {'status': 'completed', 'new_refutation_count': 0, 'new_failure_record_count': 0,
              'already_refuted_count': 0, 'already_recorded_count': 0,
              'unknown': [], 'failure_refs': [], 'new_predictions': 0, 'new_fits': 0,
              'support_updated': False, 'independent_confirmation': False}
    exports = question.get('revision_candidate_exports', [])
    if comparison.get('operator') != 'research_choice' or not exports:
        return result
    n = len(agent.rows)
    documents, locations, resolved_check_bases = {}, {}, []
    rows = json.loads(json.dumps([asdict(row) for row in agent.rows]))
    consumer = question if comparison_question is None else comparison_question
    expected_question = json.loads(json.dumps({key: consumer[key] for key in
        ('id', 'meaning_version', *SEMANTIC_FIELDS) if key in consumer}))
    target = question.get('theory_target', {})

    def load(reference):
        reference = _absolute_reference(reference, agent.workspace)
        filename = reference['file']
        if filename not in documents:
            documents[filename] = load_source_values([
                {**reference, 'json_path': []}], agent.workspace)[0]['value']
        value = documents[filename]
        for part in reference.get('json_path', []):
            value = value[part]
        return value

    for index, evaluation in enumerate(comparison.get('evaluations', [])):
        exported = [item for item in exports if item['model'] == evaluation.get('path')]
        if not exported:
            continue
        # Saved assessments use the evaluator's JSON transport: an in-memory
        # tuple history and its saved list are the same ordered paid input.
        identity = exported[0]['theory_ref']
        try:
            evaluation = json.loads(json.dumps(restore_score_history_value(evaluation, load)))
            if len(exported) != 1:
                raise ValueError('unique_revision_export_required')
            theory = agent.theory_index.get(identity['id'], {})
            if (theory.get('version') != identity['version'] or not theory.get('program')
                    or exported[0].get('program_source_ref', {}).get('version') != identity['version']):
                raise ValueError('exact_exported_Program_version_required')
            if (theory.get('context_ref') != target.get('context_ref')
                    or theory.get('output_unit') != target.get('output_unit')
                    or any(target.get('role_signature', {}).get(name) != contract
                           for name, contract in theory.get('role_signature', {}).items())):
                raise ValueError('same_revision_target_contract_required')
            mse = evaluation.get('mse')
            if (evaluation.get('returncode') != 0 or evaluation.get('truncated')
                    or evaluation.get('loss_kind') != 'mse' or evaluation.get('risk_quantile') != .5
                    or type(mse) not in (int, float) or not isfinite(mse)
                    or evaluation.get('paid_residuals', {}).get('evaluated_rows') != n
                    or comparison.get('paid_rows') != n):
                raise ValueError('complete_paid_mse_execution_required')
            execution_ref = evaluation.get('paid_score_reuse', {}).get('execution_basis_ref')
            if not execution_ref:
                raise ValueError('retained_paid_execution_reference_required')
            assessment = load(execution_ref)
            basis = load(assessment['scope_ref'])
            model = exported[0]['model']
            context = assessment['execution_context']
            reuse = evaluation.get('paid_score_reuse', {})
            alias = (assessment['model'] != model
                and reuse.get('status') == 'reused'
                and reuse.get('match_basis') == 'identical_closed_literal_execution'
                and reuse.get('matched_model') == assessment['model']
                and literal_python_model_context_matches(
                    agent.workspace, model, context['workspace'], context['model']))
            if (assessment['model'] != model and not alias) or basis['rows'] != rows:
                raise ValueError('exact_current_paid_assessment_scope_required')
            if basis['question'] != expected_question:
                raise ValueError('exact_paid_assessment_question_meaning_required')
            original = assessment['result']
            if alias and 'paid_residuals' in original:
                original = {**original, 'paid_residuals': {**original['paid_residuals'], 'model': model}}
            differences = [key for key in ('returncode', 'mse', 'loss_kind', 'risk_quantile', 'paid_residuals')
                           if original.get(key) != evaluation.get(key)]
            if differences:
                raise ValueError('exact_retained_paid_assessment_result_required:' + ','.join(differences))
            actual_source = (Path(context['workspace']) / context['model']).read_bytes()
            if actual_source != export_temporal_program(Program.from_dict(theory['program']),
                    feature_names=tuple(target['role_signature'])).encode('utf-8'):
                raise ValueError('scored_revision_Program_bytes_required')
            # A restored raw index can still contain the original relative
            # paid-row references. Its exact export ledger names the physical
            # constructor record, so use that origin rather than the new host.
            transfers = [(position, item) for position, item in enumerate(
                agent.theory_consolidation.get('transfers', [])) if item.get('theory_ref') == identity]
            binding = None
            for position, item in transfers:
                origin = library_entry_reference(agent, 'transfers', position, documents=locations)
                supplied = _anchor_file_references(theory.get('consolidation_ref'),
                                                   _reference_workspace(_absolute_reference(origin, agent.workspace)))
                if _anchor_file_references(item.get('binding'), agent.workspace) == supplied:
                    binding = supplied
                    break
            found = next(((position, item) for position, item in reversed(transfers)
                if 'check_basis' in item
                and _anchor_file_references(item.get('binding'), agent.workspace) == binding), None)
            if found is None:
                raise ValueError('exact_revision_transfer_binding_required')
            position, transfer = found
            examples = next((value for basis, value in resolved_check_bases
                             if basis == transfer['check_basis']), None)
            if examples is None:
                examples = deepcopy(transfer['check_basis'])
                references = []
                for item in examples:
                    references.extend([item['source_ref'], item['criterion']['criterion_ref']])
                # Resolve the complete same-scope batch once. Anchored source
                # criteria can address an original state much larger than a
                # row; resolving each pair separately rereads that state N times.
                for reference in references:
                    path = Path(_absolute_reference(reference, agent.workspace)['file'])
                    if path.parent == agent.workspace and path.name in {
                            'research_state.json', 'laboratory_observations.json'}:
                        reference['file'] = path.name
                values = resolve_program_sources(agent, references)
                for ri, item in enumerate(examples):
                    if values[2 * ri + 1] != item['criterion'].get('mse_threshold'):
                        raise ValueError('original_paid_mse_criterion_value_required')
                    item['observation'] = values[2 * ri]
                examples = json.loads(json.dumps(examples))
                resolved_check_bases.append((transfer['check_basis'], examples))
            criterion = examples[0].get('criterion', {}) if examples else {}
            if (len(examples) != n or criterion.get('kind') != 'paid_mse'
                    or criterion.get('evaluated_rows') != n or not criterion.get('criterion_ref')
                    or criterion.get('mse_threshold') != agent.config.confirmation_mse_threshold
                    or any(item.get('criterion') != criterion
                           or item.get('context_ref') != theory['context_ref']
                           or item.get('observation') != row for item, row in zip(examples, rows))):
                raise ValueError('original_complete_paid_mse_criterion_and_rows_required')
            if mse <= criterion['mse_threshold']:
                continue
            witness = next(((ri, saved) for ri, saved in enumerate(
                evaluation['paid_residuals'].get('examples', []))
                if type(saved.get('row_index')) is int and 0 <= saved['row_index'] < n
                and type(saved.get('squared_error')) in (int, float)
                and isfinite(saved['squared_error']) and saved['squared_error'] > n * criterion['mse_threshold']
                and all(rows[saved['row_index']].get(key, []) == saved.get(key, [])
                        for key in ('features', 'target', 'history'))), None)
            if witness is None:
                raise ValueError('saved_sufficient_single_residual_witness_unavailable')
            score_ref = _child_ref(_absolute_reference(source_ref, agent.workspace), 'evaluations', index)
            witness_ref = _child_ref(score_ref, 'paid_residuals', 'examples', witness[0])
            failure = {'evaluation_ref': score_ref,
                'assessment_ref': _child_ref(execution_ref, 'result'),
                'execution_basis_ref': deepcopy(execution_ref), 'scope_ref': deepcopy(assessment['scope_ref']),
                'witness_ref': witness_ref, 'paid_row_ref': deepcopy(examples[witness[1]['row_index']]['source_ref']),
                'criterion_ref': deepcopy(criterion['criterion_ref']), 'evaluated_rows': n,
                'sufficiency': 'one_saved_residual_proves_this_instance_paid_mse_failure_only'}
            if any(item.get('ordinary_comparison_failure', {}).get('assessment_ref') == failure['assessment_ref']
                   and _anchor_file_references(item.get('binding'), agent.workspace) == binding
                   for _, item in transfers):
                result['already_recorded_count'] += 1
                continue
            # Keep the old record and every unexecuted point check. This new
            # aggregate fact is evidence about the same Program version, rather
            # than a new executable version requiring another export/score.
            prior_ref = library_entry_reference(agent, 'transfers', position, documents=locations)
            record = {'theory_id': identity['id'], 'theory_ref': deepcopy(identity),
                'binding': deepcopy(transfer['binding']), 'status': 'refuted',
                'ordinary_comparison_failure': failure, 'prior_transfer_ref': prior_ref,
                'checks_ref': _child_ref(prior_ref, 'checks'),
                'check_basis_ref': _child_ref(prior_ref, 'check_basis'),
                'support_updated': False, 'independent_confirmation': False,
                'paid_mse_check': {**deepcopy(criterion), 'status': 'failed', 'mse': mse,
                    'scope': 'retained_paid_evaluation', 'source_ref': score_ref,
                    'independent_confirmation': False},
                'retained_counterexample_refs': [score_ref, witness_ref]}
            agent.theory_consolidation['transfers'].append(record)
            was_refuted = (theory.get('compatible') is False
                           or theory.get('support', {}).get('status') == 'refuted')
            theory['compatible'] = False
            theory.setdefault('support', {})['status'] = 'refuted'
            # Persist the now-established physical constructor origin too;
            # snapshotting a restored relative binding at the new host would
            # otherwise redirect its old paid-row references.
            theory['consolidation_ref'] = deepcopy(binding)
            result['new_failure_record_count'] += 1
            result['already_refuted_count' if was_refuted else 'new_refutation_count'] += 1
            result['failure_refs'].append({'theory_ref': deepcopy(identity), 'evaluation_ref': score_ref})
        except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
            result['unknown'].append({'theory_ref': deepcopy(identity), 'reason': str(error)})
    if result['new_failure_record_count']:
        save_theory_library(agent)
        save_theory_index(agent)
    if result['unknown']:
        result['status'] = 'UNKNOWN'
    return result


def retained_checkpoint_result(previous, state):
    """Use committed scientific state with result-only paid metadata.

    The caller transfers the already decoded state to the restored agent. A
    partially saved compact training result must not replace newer paid work.
    """
    if any(state['costs'][key] < value for key, value in previous['costs'].items()):
        raise ValueError('research state account precedes its training result')
    for key in ('observation_cost', 'measurement_cost', 'model_calls'):
        if state['costs'][key] != previous['costs'][key]:
            raise ValueError('research state requires paid acquisitions absent from training')
    current = dict(previous)
    current.update({key: state[key] for key in (
        'costs', 'evaluated_model', 'method_results', 'program_contexts',
        'experiment_comparisons', 'observable_models', 'conditional_models',
        'model_dependencies', 'research_candidates', 'research_input_bindings',
        'relation_research', 'theory_index', 'research_mission', 'research_agenda',
        'theory_consolidation', 'research_organization') if key in state})
    current['online_questions'] = state['questions']
    current['revision_proposals'] = state['revisions']
    return current


class CommonArchiveInquiryAgent(ProcedureConsumerAgent):
    """Use the old Python/procedure loop with equal executable persistence."""

    common_interface = (COMMON_ARCHIVE_INTERFACE + ONLINE_RESEARCH_INTERFACE
                        + OBSERVABLE_INTERFACE + CONDITIONAL_INTERFACE + EVIDENCE_INTERFACE
                        + STRUCTURAL_RESEARCH_INTERFACE + NATIVE_CONDITION_INTERFACE + QUESTION_INTERFACE
                        + ROUTING_INTERFACE + CONSOLIDATION_INTERFACE)

    def __init__(self, *args, **kwargs):
        if kwargs.get("arm") not in {"code_react", "hegel"}:
            raise ValueError("common archive conditions are code_react and hegel")
        super().__init__(*args, **kwargs)
        self.experiment_comparisons = []
        self.observable_models = {}
        self.conditional_models = {}
        self.model_dependencies = []
        self.research_input_bindings = {}
        self.research_candidates = {}
        self.relation_research = {}
        self.theory_index = install_theory_index(self, self.knowledge.get('theory_index'))
        self.research_mission = {}
        self.research_agenda = {}
        self.theory_consolidation = install_theory_library(self, self.knowledge.get('theory_consolidation'))
        if self.knowledge.get('theory_consolidation'):
            save_theory_library(self)
        self.llm_inquiries = {}
        self._inquiry_stack = []
        self._active_question_id = None
        if self.arm == "hegel":
            root = self._open_question({"id": "task", "question":
                "Infer the process from current paid observations and predict new public trajectories.",
                "parent_id": None, "depends_on": []})
            root["origin"] = "public_task"
            self._active_question_id = root["id"]

    def restore_research(self, previous, previous_workspace, laboratory, *, confirmation_seed,
                         retained_comparison_reports=(), retained_state=None):
        """Continue saved fixed-apparatus research, preserving its paid account.

        Construct this agent in a new workspace with the intended total limits.
        Only the original task's files and observations enter the continuation.
        """
        context = load_code_context(previous_workspace)
        self.knowledge = deepcopy(context["session_knowledge"])
        self.source_task = context["research_phase"] == "source_learning"
        return self._restore_paid_research(previous, previous_workspace, laboratory,
            confirmation_seed=confirmation_seed, context=context,
            retained_comparison_reports=retained_comparison_reports,
            retained_state=retained_state)

    def restore_practice_research(self, previous, previous_workspace, laboratory, *, confirmation_seed,
                                  retained_comparison_reports=(), retained_state=None):
        """Continue a zero-LLM practice directly from its actual saved result/state.

        The caller supplies the original knowledge and new total limits. No online
        request is fabricated for a frozen practice that never made an LLM call.
        """
        return self._restore_paid_research(previous, previous_workspace, laboratory,
            confirmation_seed=confirmation_seed,
            retained_comparison_reports=retained_comparison_reports,
            retained_state=retained_state)

    def _restore_paid_research(self, previous, previous_workspace, laboratory, *,
                               confirmation_seed, context=None, retained_comparison_reports=(),
                               retained_state=None):
        source = Path(previous_workspace).resolve()
        if source == self.workspace:
            raise ValueError("restore into a separate workspace to retain the original run")
        if laboratory.costs["observation_cost"]:
            raise ValueError("restore with a fresh apparatus; do not repurchase the prefix")
        if getattr(laboratory, "_settings", {}).get("calibration_change") is not None:
            raise ValueError("this continuation restores a fixed apparatus")
        state = (read_json_checkpoint(source / "research_state.json")
                 if retained_state is None else retained_state)
        retain = deepcopy
        if retained_state is not None:
            previous = retained_checkpoint_result(previous, state)
            retain = lambda value: value
        questions = retain(previous["online_questions"])
        for question in questions:
            reference = question.get("native_condition_research")
            if (isinstance(reference, dict)
                    and reference.get("full_content_file") == "research_state.json"):
                rounds = state
                for key in reference["json_path"]:
                    rounds = rounds[key]
                question["native_condition_research"] = deepcopy(rounds[:reference["round_count"]])
        self.inherited_files = set(previous["inherited_files"])
        contexts = {}
        comparison_reports, retained_parameters, omitted_parameters = {}, set(), set()

        def retain_parameters(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == 'parameter_path' and isinstance(item, str) and item:
                        original = (source / item).resolve()
                        if original.is_file():
                            value[key] = str(original)
                            retained_parameters.add(original)
                            if original.is_relative_to(source):
                                omitted_parameters.add(original)
                    elif isinstance(item, (dict, list)):
                        retain_parameters(item)
            elif isinstance(value, list):
                for item in value:
                    retain_parameters(item)

        # Only explicitly declared, completed comparisons become historical
        # evidence. Their report/residuals stay local; fitted fold parameters
        # remain at their retained original paths, never writable links here.
        # A later fit can create the same local filename as an independent file.
        for name in retained_comparison_reports:
            path = source / name
            if not path.is_file():
                continue
            report = json.loads(path.read_text(encoding='utf-8'))
            if report.get('complete') is True:
                retain_parameters(report)
                comparison_reports[name] = report

        def ignored(directory, names):
            root = Path(directory)
            excluded = set()
            for name in names:
                path = root / name
                if omitted_parameters and path.resolve() in omitted_parameters:
                    excluded.add(name)
                elif name == '_evaluated_model' and path.is_dir():
                    contexts.setdefault(path, set())
                    excluded.add(name)
                elif _archived_paid_trace(path) and path.is_symlink():
                    # The constructor materialized inherited files already.
                    # Replace that copy with the preserved immutable reference.
                    (self.workspace / path.relative_to(source)).unlink(missing_ok=True)
                elif (_model_context_ignored(path, source)
                      or (root == source and name in {
                          "research_trace.json", "training.json.writing"})):
                    excluded.add(name)
            return excluded

        # Stop at each primary execution root. Its compact copy below retains
        # fitting inputs without recursively walking copies of earlier refits.
        excluded = ignored(source, [path.name for path in source.iterdir()])
        for path in source.iterdir():
            if path.name in excluded:
                continue
            target = self.workspace / path.name
            if path.name == "_submitted_models" and path.is_dir():
                # Past submissions are evidence, not writable current models.
                # Keep the parent local so later confirmations get new roots.
                target.mkdir(parents=True, exist_ok=True)
                for submission in path.iterdir():
                    retained = target / submission.name
                    if not retained.exists():
                        retained.symlink_to(submission.resolve(), target_is_directory=True)
                continue
            if path.is_dir():
                shutil.copytree(path, target, dirs_exist_ok=True, symlinks=True,
                                ignore=ignored, copy_function=_copy_model_file)
            else:
                _copy_model_file(path, target)

        # Authored programs read ordinary complete JSON. Seed the state from
        # its retained predecessor so the next atomic write can preserve
        # unchanged filesystem blocks instead of rebuilding the history.
        # The local state remains independently writable.
        _copy_model_file(source / "research_state.json", self.workspace / "research_state.json")
        for name in ('research_state.json', 'active.json', 'training.json'):
            retain_checkpoint_sources(self.workspace / name,
                source / 'research_state.json', source / 'training.json')

        def referenced_paths(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {'model', 'source_model', 'tested_model', 'executed_path',
                               'path', 'view', 'source_view', 'returned_model'}:
                        if isinstance(item, str):
                            yield item
                    elif key == 'artifacts' and isinstance(item, list):
                        yield from (name for name in item if isinstance(name, str))
                    if (isinstance(item, (dict, list)) and key not in {
                            'files', 'source', 'rows', 'bound_rows', 'observations',
                            'paid_observations', 'predictions', 'prediction_inputs',
                            'history', 'features'}):
                        yield from referenced_paths(item)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (dict, list)):
                        yield from referenced_paths(item)

        # A directly selected model can live deeper than a primary root. Read
        # actual path fields only; errors, notes and source strings are not paths.
        references = set(state['exported_candidate_paths'])
        references.update(previous.get('research_candidates', {}))
        if previous.get('model'):
            references.add(previous['model'])
        metadata = [state['operator_results'], previous['events'], questions,
                    previous.get('evaluated_model'), previous['outgoing_knowledge'].get('manifest')]
        metadata.extend(previous.get(field, {}) for field in
                        ('observable_models', 'conditional_models', 'relation_research',
                         'research_candidates', 'model_dependencies'))
        for value in metadata:
            references.update(referenced_paths(value))
        for name in references:
            if Path(name).parts[:1] == ("_submitted_models",):
                continue
            if '_evaluated_model' in Path(name).parts and (source / name).is_file():
                context_root, inner = python_model_context(source, name)
                contexts.setdefault(context_root, set()).add(inner)
        bindings_file = source / 'model_bindings.json'
        if bindings_file.is_file():
            for model in json.loads(bindings_file.read_text()):
                for context_root, inner in _bound_model_contexts(source, model):
                    if context_root != source:
                        contexts.setdefault(context_root, set()).add(inner)
        for context_root in sorted(contexts, key=lambda path: len(path.parts)):
            # External historical roots keep their saved absolute references.
            if not context_root.is_relative_to(source):
                continue
            if context_root.is_relative_to(source / "_submitted_models"):
                continue
            bindings_file = context_root / 'model_bindings.json'
            models = contexts[context_root]
            if bindings_file.is_file():
                models.update(json.loads(bindings_file.read_text()))
            _copy_execution_context(context_root,
                self.workspace / context_root.relative_to(source), models)
        for name, report in comparison_reports.items():
            write_text_incrementally(self.workspace / name, json.dumps(report, indent=2))
        # The last request precedes the last executed action. Refresh its files
        # from disk and add successful final writes, not another arm's archive.
        names = set(context["files"] if context is not None else self.files)
        names.update(previous.get("outgoing_knowledge", {}).get("files", {}))
        if context is None:
            # The frozen method can write predictors through its Python steps;
            # expose those real source files without importing a nonexistent request.
            names.update(str(path.relative_to(self.workspace))
                         for path in self.workspace.glob("*.py"))
            if previous.get("method_path"):
                names.add(previous["method_path"])
        names.update(event.get("feedback", {}).get("written") for event in previous["events"]
                     if event.get("feedback", {}).get("written"))
        self.files = {name: (self.workspace / name).read_text(encoding="utf-8")
                      for name in names if "_evaluated_model" not in Path(name).parts
                      and not _model_context_ignored(self.workspace / name, self.workspace)
                      and (self.workspace / name).is_file()}
        self.events = retain_event_history(previous["events"])
        self.observations = deepcopy(previous["paid_observations"])
        self.rows = [row for record in self.observations for row in laboratory.observation_rows(record)]
        self.costs = deepcopy(previous["costs"])
        self.model = previous["model"]
        self.questions = questions
        self.revisions = retain(previous["revision_proposals"])
        self.experiment_comparisons = retain(previous["experiment_comparisons"])
        self.observable_models = retain(previous.get("observable_models", {}))
        self.conditional_models = retain(previous.get("conditional_models", {}))
        self.model_dependencies = retain(previous.get("model_dependencies", []))
        self.research_input_bindings = retain(previous.get("research_input_bindings", {}))
        self.research_candidates = retain(previous.get("research_candidates", {}))
        self.relation_research = retain(previous.get("relation_research", {}))
        install_theory_index(self, self.knowledge.get('theory_index'))
        self.theory_index = retain(previous.get('theory_index', state.get('theory_index', {})))
        self.research_mission = retain(previous.get('research_mission', state.get('research_mission', {})))
        self.research_agenda = retain(previous.get('research_agenda', state.get('research_agenda', {})))
        library = source / LIBRARY_PATH
        if library.is_file():
            self._save(LIBRARY_PATH, library.read_text())
            self.theory_consolidation = install_theory_library(self)
        self.llm_inquiries = deepcopy(previous.get('llm_inquiries', {}))
        self._inquiry_stack = deepcopy(previous.get('inquiry_stack', []))
        for frame in self._inquiry_stack:
            frame['question'] = next(q for q in self.questions if q['id'] == frame['question_id'])
        self.method_results = retain(previous["method_results"])
        self.program_contexts = retain(previous.get("program_contexts", {}))
        self.operator_results = retain(state["operator_results"])
        self.candidate_paths = retain(state["exported_candidate_paths"])
        self.manifest = deepcopy(previous["outgoing_knowledge"].get("manifest", {}))
        self._active_question_id = (context["online_research"]["active_question"]
            if context is not None and retained_state is None
            else state["online_research"]["active_question"])
        question_ids = {question["id"] for question in self.questions}
        for event in self.events:
            args, feedback = event.get("arguments", {}), event.get("feedback", {})
            declaration = args.get("research_question") or args.get("question") or feedback.get("question")
            identifier = feedback.get("research_question_id") or (
                declaration.get("id") if isinstance(declaration, dict) else None)
            if identifier in question_ids and retained_state is None:
                self._active_question_id = identifier
        retained = previous["evaluated_model"]
        self.evaluated_model = None if retained is None else {
            "model": str(Path(retained["model"]).relative_to("_evaluated_model")),
            "source_model": retained["source_model"], "current_mse": retained["mse"],
            "evaluated_rows": retained["evaluated_rows"]}
        if retained is not None:
            self.evaluated_model.update({key: deepcopy(retained[key]) for key in
                ('evaluation_scope', 'fitting_view', 'later_readings', 'selection') if key in retained})
        self.confirmations = []

        def collect_confirmations(value):
            if isinstance(value, dict):
                confirmation = value.get("fresh_confirmation")
                if confirmation and "observations" in confirmation:
                    indices = confirmation["observation_indices"]
                    if not any(item["observation_indices"] == indices for item in self.confirmations):
                        self.confirmations.append(deepcopy(confirmation))
                for key, item in value.items():
                    if key != "fresh_confirmation":
                        collect_confirmations(item)
            elif isinstance(value, list):
                for item in value:
                    collect_confirmations(item)

        for event in self.events:
            collect_confirmations(event.get("feedback", {}))
        # The result also carries the last confirmation directly. Event records
        # are normally complete; de-duplication avoids consuming the stream twice.
        collect_confirmations({"fresh_confirmation": previous.get("fresh_confirmation")})
        self.confirmation_rng = Random(confirmation_seed)
        for confirmation in self.confirmations:
            actions = laboratory.confirmation_actions(confirmation["rows"], self.confirmation_rng)
            if actions != [record["action"] for record in confirmation["observations"]]:
                raise ValueError("confirmation stream differs from the saved paid acquisitions")
        # Every temporal trajectory resets physical state. Carry its already
        # paid account forward without purchasing or simulating it again.
        restore_account = getattr(laboratory, "restore_paid_account", None)
        if restore_account is not None:
            restore_account(self.observations, self.costs)
        else:
            account = {"observation_cost": self.costs["observation_cost"],
                "measurement_cost": self.costs["measurement_cost"], "queries": len(self.observations)}
            if isinstance(laboratory, AnonymousHistoryLab):
                laboratory._costs = account
            else:
                laboratory._past_costs = account
        if self.provider is not None:
            self.provider.calls = self.costs["model_calls"]
        restore_comparison_history(self)
        self._restore_scientific_returns()
        self._write_evidence()
        self._prepare_program_state()
        restored = {"restored_paid_rows": len(self.rows), "costs_before": deepcopy(self.costs),
                    "restored_confirmations": len(self.confirmations), "new_observations": 0}
        if retained_comparison_reports:
            restored['completed_comparison_storage'] = {
                'reports': list(comparison_reports),
                'retained_parameter_files': sorted(str(path) for path in retained_parameters),
                'parameter_bytes_not_copied': sum(path.stat().st_size for path in omitted_parameters),
            }
        return restored

    def _start_inquiry(self, arguments, *, parent_event_index=None):
        if self.provider is None:
            raise ValueError('inquire needs the live model provider, not a frozen zero-LLM replay')
        calls = arguments.get('max_model_calls', 8)
        if type(calls) is not int or calls < 1:
            raise ValueError('max_model_calls must be a positive integer')
        declaration = deepcopy(arguments['question'])
        if any(frame['question_id'] == declaration['id'] for frame in self._inquiry_stack):
            raise ValueError('continue the active inquiry instead of delegating it to itself')
        if self._inquiry_stack:
            self.llm_inquiries[self._inquiry_stack[-1]['question_id']]['notes'] = (
                self.workspace / 'research_notes.md').read_text()
        declaration.setdefault('parent_id', self._active_question_id)
        parent_question = (self._inquiry_stack[-1]['question_id'] if self._inquiry_stack
                           else declaration.get('parent_id'))
        question = self._open_question(declaration)
        if question.get('status') in {'supported', 'refuted'}:
            # Explicit continuation keeps the old answer while reopening its research.
            question['status'] = 'reopened'
        session, frame = begin_session(self.llm_inquiries, self._inquiry_stack, question,
            start_calls=self.costs['model_calls'], calls=calls,
            total_calls=self.config.max_model_calls, assignment=arguments.get('assignment'))
        frame['call_limit'] = min(frame['call_limit'], self.config.max_model_calls - 1)
        if self._inquiry_stack:
            frame['call_limit'] = min(frame['call_limit'], self._inquiry_stack[-1]['call_limit'] - 1)
        parent_event_index = len(self.events) if parent_event_index is None else parent_event_index
        frame.update(parent_event_index=parent_event_index, parent_question=parent_question,
                     awaiting_first_turn=True, event_start=max(parent_event_index + 1, len(self.events)),
                     event_indices=[], artifacts=[])
        session.setdefault('event_indices', [])
        enter_notebook(self.workspace, session, frame)
        self.files['research_notes.md'] = session['notes']
        self._inquiry_stack.append(frame)
        self._active_question_id = question['id']
        return {'operator': 'inquire', 'status': 'inquiry_started',
                'question': deepcopy(question), 'assignment': deepcopy(frame['assignment'])}

    def _inquiry_feedback(self):
        """Record completed child turns, including a recovered paid Python action."""
        for frame in self._inquiry_stack:
            session = self.llm_inquiries[frame['question_id']]
            start = (frame['event_indices'][-1] + 1 if frame['event_indices'] else frame['event_start'])
            for index in range(start, len(self.events)):
                frame['event_indices'].append(index)
                session['event_indices'].append(index)
                event = self.events[index]
                feedback, args = event.get('feedback', {}), event.get('arguments', {})
                declared = args.get('artifacts', [])
                paths = [feedback.get('written'), args.get('model'),
                         *(declared if isinstance(declared, list) else [])]
                for name in paths:
                    if isinstance(name, str) and (self.workspace / name).is_file() and name not in frame['artifacts']:
                        frame['artifacts'].append(name)
        if self._inquiry_stack:
            notes = self.workspace / 'research_notes.md'
            if notes.exists():
                self.llm_inquiries[self._inquiry_stack[-1]['question_id']]['notes'] = notes.read_text()

    def _return_inquiry(self, status):
        frame = self._inquiry_stack.pop()
        session = self.llm_inquiries[frame['question_id']]
        calls = self.costs['model_calls'] - frame['start_calls']
        session['model_calls'] += calls
        result = {'operator': 'inquire', 'status': status,
                  'question': deepcopy(frame['question']), 'assignment': deepcopy(frame['assignment']),
                  'parent_question': frame['parent_question'],
                  'model_calls': calls, 'notes': session['notes'],
                  'artifacts': list(frame['artifacts']), 'event_indices': list(frame['event_indices'])}
        self._record_inquiry_return(frame, result)
        leave_notebook(self.workspace, session, frame, status)
        notes = self.workspace / 'research_notes.md'
        if notes.exists():
            self.files['research_notes.md'] = notes.read_text()
        else:
            self.files.pop('research_notes.md', None)
        self._active_question_id = frame['parent_question']

    def _record_inquiry_return(self, frame, result):
        # Complete the paid parent delegation in place. Child decisions already
        # occupy their own chronological events and are never charged twice.
        event = self.events[frame['parent_event_index']]
        if event.get('arguments', {}).get('operator') == 'inquire':
            event['feedback'] = deepcopy(result)
        else:
            event['feedback']['inquiry_result'] = deepcopy(result)
        if self.events and self.events[-1] is not event:
            self.events[-1].setdefault('feedback', {})['inquiry_return'] = deepcopy(result)
        question = next((q for q in self.questions if q['id'] == result['question']['id']), None)
        if question is not None and question.get('origin') == 'fresh_confirmation_failure':
            question['review_return'] = {key: deepcopy(result[key]) for key in
                                         ('status', 'model_calls', 'artifacts', 'event_indices')}
        record_representation_return(self, frame, result)
        record_structural_return(self, frame, result)
        self._record_scientific_return(result, parent_id=frame['parent_question'])

    def _record_scientific_return(self, result, *, parent_id=None):
        """Index a generic child's executed checks or evaluated delivery for its caller."""
        identifier = result['question']['id']
        question = next(q for q in self.questions if q['id'] == identifier)
        if question.get('origin') in {'host_execution_repair', 'fresh_confirmation_failure',
                                     'representation_research', 'structural_proposal'}:
            return
        indices = result['event_indices']
        revisions = []
        for index in indices:
            feedback = self.events[index].get('feedback', {})
            records = [feedback.get('revision'), feedback.get('evidence_readout_trial')]
            for record in records:
                if not isinstance(record, dict):
                    continue
                revision_index = record.get('index', record.get('revision_index'))
                if (revision_index is not None and revision_index not in revisions
                        and self.revisions[revision_index].get('analysis_attempts')):
                    revisions.append(revision_index)
        evaluation = extract_existing_return_evaluation(self, indices)
        if not any(self.revisions[index].get('check_results') for index in revisions) and evaluation is None:
            return
        assignment = result.get('assignment', result['question'])
        parent_id = (parent_id if parent_id is not None else
                     result.get('parent_question', assignment.get('parent_id')))
        if parent_id is None:
            return
        returned = {key: deepcopy(result[key]) for key in
                    ('status', 'model_calls', 'artifacts', 'event_indices')}
        returned.update(parent_id=parent_id, question_id=identifier,
            assignment={key: deepcopy(assignment.get(key)) for key in
                        ('id', 'question', 'parent_id', 'depends_on')},
            revision_indices=revisions)
        if evaluation is not None:
            returned['evaluation'] = evaluation
        previous = question.get('research_return')
        if previous is None or max(indices, default=-1) >= max(previous['event_indices'], default=-1):
            question['research_return'] = returned

    def _restore_scientific_returns(self):
        """Recover return references from saved events without replaying paid work."""
        for event in self.events:
            feedback = event.get('feedback', {})
            returns = [feedback.get('inquiry_result'), feedback.get('inquiry_return')]
            if feedback.get('operator') == 'inquire' and 'event_indices' in feedback:
                returns.append(feedback)
            for returned in returns:
                if isinstance(returned, dict) and returned.get('operator') == 'inquire':
                    self._record_scientific_return(returned)

    def _settle_inquiries(self):
        self._inquiry_feedback()
        while self._inquiry_stack:
            frame = self._inquiry_stack[-1]
            question = frame['question']
            same_assignment = all(question.get(key) == frame['assignment'].get(key)
                                  for key in ('question', 'parent_id', 'depends_on'))
            states = {q['id']: q.get('status') for q in self.questions}
            pending = any(states.get(identifier) != 'supported'
                          for identifier in question.get('depends_on', []))
            if frame['stop_requested']:
                status = 'inquiry_finished'
            elif (question.get('origin') != 'structural_proposal' and same_assignment
                  and question.get('status') in {'supported', 'refuted'} and not pending):
                status = 'answered'
            elif self.costs['model_calls'] >= frame['call_limit']:
                status = 'model_call_budget_exhausted'
            else:
                break
            self._return_inquiry(status)

    def _before_model_turn(self, laboratory):
        self._settle_inquiries()
        opened = self._open_scientific_revision(laboratory)
        opened = self._prepare_parent_research() or opened
        opened = open_structural_proposal(self) or opened
        opened = open_representation_inquiry(self) or opened
        if self._inquiry_stack:
            self._inquiry_stack[-1]['awaiting_first_turn'] = False
            self._active_question_id = self._inquiry_stack[-1]['question_id']
        return opened

    def _after_model_turn(self, laboratory):
        was_child = bool(self._inquiry_stack)
        self._settle_inquiries()
        if not was_child:
            self._repair_repeated_execution_error()

    def _open_scientific_revision(self, laboratory):
        """Give an actual failed fresh prediction an executable research obligation."""
        if (self.arm != 'hegel' or self.provider is None or self._inquiry_stack
                or self.model is not None or not self.confirmations
                or self.config.max_model_calls - self.costs['model_calls'] <= 1
                or self.config.observation_budget - self.costs['observation_cost'] < self.config.confirmation_rows
                or self.costs['python_execution_seconds'] >= self.config.execution_seconds
                or self.costs['reported_tokens'] is None
                or self.costs['reported_tokens'] >= self.config.token_budget):
            return
        confirmation = self.confirmations[-1]
        mse = confirmation.get('mse')
        if (confirmation.get('passed') is not False or confirmation.get('returncode') != 0
                or confirmation.get('timeout') or not isinstance(mse, (float, int))
                or not isfinite(mse)):
            return
        indices = confirmation['observation_indices']
        if any(q.get('origin') == 'fresh_confirmation_failure'
               and q.get('confirmation_observation_indices') == indices for q in self.questions):
            return

        def contains(value):
            if isinstance(value, dict):
                fresh = value.get('fresh_confirmation')
                if isinstance(fresh, dict) and fresh.get('observation_indices') == indices:
                    return True
                return any(contains(item) for key, item in value.items() if key != 'fresh_confirmation')
            return isinstance(value, list) and any(contains(item) for item in value)

        event_index = next((index for index in range(len(self.events) - 1, -1, -1)
                            if contains(self.events[index].get('feedback', {}))), None)
        if event_index is None:
            return
        number = len(self.confirmations) - 1
        question = self._open_question({
            'id': f'scientific_revision_{number}', 'parent_id': 'task', 'depends_on': [],
            'origin': 'fresh_confirmation_failure',
            'confirmation_observation_indices': deepcopy(indices),
            'failed_confirmation': {'confirmation_index': number,
                **{key: deepcopy(confirmation[key]) for key in
                   ('passed', 'returncode', 'mse', 'mse_threshold', 'rows', 'observation_indices',
                    'submitted_candidate', 'residuals')
                   if key in confirmation}},
            'question': 'Investigate the submitted predictor\'s failed independent confirmation. '
                'Use its frozen candidate and actual counterexamples to determine what remains '
                'unexplained and what evidence could distinguish parameter error from an inadequate '
                'representation or mechanism. This failure refutes that submitted predictor, not '
                'every member of its family. Choose and test your own explanation using the paid '
                'records and public tools. Return an executable analysis, component or candidate '
                'with its measured support and limitations, so the parent can evaluate and adopt '
                'or reject it. Declare any parameter or component dependencies through the existing '
                'binding tools when your parent candidate uses them. Local finish returns this '
                'research; only a later parent submission obtains independent confirmation.'})
        failures = generalization_failures(self.confirmations, self.config.confirmation_mse_threshold)
        if len(failures) >= 2:
            question.update(review_scope='prediction_method_revision', review_model_calls=12,
                generalization_review={'failed_attempts': failures,
                    'evidence_scope': 'Frozen submitted predictors and their actual independent '
                        'prediction failures on this paid task; no whole-family refutation.'})
            question['question'] = PREDICTION_METHOD_QUESTION
            question['assignment']['question'] = question['question']
            prepare_generalization_review(self, question, confirmation, laboratory)
            self.events[event_index]['feedback']['generalization_review'] = {
                'question_id': question['id'], 'review_scope': question['review_scope'],
                'generalization_review': deepcopy(question['generalization_review']),
                'whole_trajectory_refit': deepcopy(question['whole_trajectory_refit'])}
        self._dispatch_scientific_revision(question, event_index)
        return True

    def _dispatch_scientific_revision(self, question, parent_event_index):
        """The matched flat study overrides only this review's organization."""
        feedback = self._start_inquiry({'question': question,
                                       'max_model_calls': question.get('review_model_calls', 8)},
                                       parent_event_index=parent_event_index)
        self.events[parent_event_index]['feedback']['scientific_revision'] = feedback

    def _scientific_revision_assignment(self):
        question = next((q for q in reversed(self.questions)
                         if q.get('origin') == 'fresh_confirmation_failure'), None)
        if question is None:
            return None
        return {**deepcopy(question['assignment']), 'origin': question['origin'],
                'failed_confirmation': deepcopy(question['failed_confirmation']),
                **{key: deepcopy(question[key]) for key in
                   ('review_scope', 'generalization_review', 'whole_trajectory_refit')
                   if key in question}}

    def _parent_research(self):
        if self._inquiry_stack or any(q.get('flat_review', {}).get('status') == 'running'
                                     for q in self.questions):
            return None
        return next((q for q in reversed(self.questions)
                     if q.get('parent_research', {}).get('status') == 'running'), None)

    def _refresh_parent_research_phase(self, owner):
        """Route the current candidate using its actual paid evaluation and mapping."""
        research_repairs(self)
        focus = owner['parent_research']
        model = focus.get('evaluated_model')
        candidate = self.research_candidates.get(model, {})
        evaluation = candidate.get('current_evaluation', {})
        diagnostics = [evaluation[key] for key in
                       ('representation_diagnostic', 'training_representation_diagnostic')
                       if isinstance(evaluation.get(key), dict)
                       and evaluation[key].get('empirical_mse_floor') is not None]
        diagnostic = max(diagnostics, key=lambda item: item['empirical_mse_floor'], default=None)
        current = (candidate.get('paid_rows') == len(self.rows)
                   and candidate.get('status') not in {'needs_rebuild', 'needs_refit',
                                                       'new_paid_rows_unscored'})
        mse = evaluation.get('mse')
        method_result = focus.get('method_review_result', {})
        if method_result.get('source_model') != model:
            method_result = owner.get('whole_trajectory_refit', {})
        method_revision = (owner.get('review_scope') == 'prediction_method_revision'
            and method_result.get('status') != 'refit_supported_on_validation')
        validated = (current and evaluation.get('evaluation_scope') == 'paid_trajectory_holdout'
            and evaluation.get('returncode') == 0 and isinstance(mse, (int, float))
            and isfinite(mse) and mse <= self.config.confirmation_mse_threshold)
        if owner.get('review_scope') == 'prediction_method_revision':
            focus['method_review_result'] = {
                **{key: deepcopy(method_result[key]) for key in
                   ('status', 'source_model', 'tested_model', 'error', 'validation_returncode',
                    'train_rows', 'holdout_rows', 'validation_records', 'paid_rows_at_refit')
                   if key in method_result},
                'validation_mse': method_result.get('validation_mse',
                    method_result.get('validation', {}).get('mse'))}
        if diagnostic and diagnostic['empirical_mse_floor'] > self.config.confirmation_mse_threshold:
            phase = 'representation_conflict'
        elif method_revision and not validated:
            phase = 'prediction_method_revision'
        elif (current and evaluation.get('returncode') == 0
              and isinstance(mse, (int, float)) and isfinite(mse)
              and mse <= self.config.confirmation_mse_threshold):
            phase = 'awaiting_independent_confirmation'
        elif evaluation.get('returncode') == 0 and diagnostic is None:
            phase = 'representation_unassessed'
        else:
            phase = 'candidate_revision'
        focus.update(phase=phase, evaluation_current=current,
                     candidate_paid_rows=candidate.get('paid_rows'), current_paid_rows=len(self.rows))
        representation = next((q for q in self.questions
            if q['id'] == f'representation:{model}'
            and q.get('origin') == 'computed_representation_conflict'
            and q.get('status') == 'open'), None)
        self._active_question_id = representation['id'] if representation else focus['parent_id']
        return focus

    def _prepare_parent_research(self):
        """Continue an actual child return until its executable consequences are tested."""
        if (self.arm != 'hegel' or self.provider is None or self.model is not None
                or self._inquiry_stack or any(q.get('flat_review', {}).get('status') == 'running'
                                             for q in self.questions)
                or self.costs['model_calls'] >= self.config.max_model_calls
                or self.costs['python_execution_seconds'] >= self.config.execution_seconds
                or self.costs['reported_tokens'] is None
                or self.costs['reported_tokens'] >= self.config.token_budget
                or self.config.observation_budget - self.costs['observation_cost']
                    < self.config.confirmation_rows):
            return
        # Actual new child work also reopens its caller's synthesis. Older
        # failed-confirmation reviews keep their own result and disposition.
        by_id = {question['id']: question for question in self.questions}
        returns = [(q, q['review_return']) for q in self.questions
                   if q.get('origin') == 'fresh_confirmation_failure' and q.get('review_return')]
        returns.extend((by_id[returned['parent_id']], returned)
                       for q in self.questions if (returned := q.get('research_return'))
                       and returned['parent_id'] in by_id)
        if not returns:
            return
        question, summary = max(returns, key=lambda item: max(item[1]['event_indices'], default=-1))
        returned = summary['event_indices']
        active = self._parent_research()
        if active is question and active['parent_research']['returned_event_indices'] == returned:
            self._refresh_parent_research_phase(active)
            return
        if active is not None and active is not question:
            active['parent_research'].update(status='continued_by_later_review',
                                             next_review=question['id'])
        previous = question.get('parent_research')
        if previous is not None and previous['returned_event_indices'] == returned:
            # These real saved runs recorded delivery, not scientific success.
            # Resume their original work and accounting instead of suppressing
            # the unresolved parent forever because its return was seen once.
            if previous['status'] == 'parent_evaluated':
                question.setdefault('parent_research_history', []).append(deepcopy(previous))
                previous['status'] = 'running'
                previous['reopened_at_model_call'] = self.costs['model_calls']
                previous.pop('completed_at_model_call', None)
                self._refresh_parent_research_phase(question)
                return True
            return
        if previous is not None:
            question.setdefault('parent_research_history', []).append(deepcopy(previous))
        generic_return = 'question_id' in summary
        if not generic_return:
            evaluated = (question['failed_confirmation'].get('submitted_candidate')
                         or self._evaluated_summary())
        else:
            if question.get('origin') != 'fresh_confirmation_failure':
                question['review_return'] = deepcopy(summary)
            evaluated = self._evaluated_summary()
        path = evaluated.get('model') if evaluated else None
        source = (self.workspace / path).read_text() if path and (self.workspace / path).is_file() else None
        question['parent_research'] = {
            'status': 'running', 'returned_event_indices': deepcopy(returned),
            'parent_id': (summary['parent_id'] if generic_return else
                          question['assignment'].get('parent_id') or 'task'),
            'returned_research': deepcopy(summary),
            'started_at_model_call': self.costs['model_calls'],
            'baseline_parent': deepcopy(evaluated), 'baseline_parent_source': source,
            'executions': [],
        }
        self._refresh_parent_research_phase(question)
        return True

    def _record_parent_research(self, owner, action, arguments, result):
        if owner is None or owner['parent_research']['status'] != 'running':
            return
        focus = owner['parent_research']
        if action == 'finish' and result.get('fresh_confirmation', {}).get('passed') is True:
            focus.update(status='independent_confirmation_passed',
                         completed_at_model_call=self.costs['model_calls'])
            return
        operator = arguments.get('operator')
        if action not in {'evaluate', 'finish'} and not (action == 'run' and operator in {
                'bind_research', 'analyze_evidence', 'analyze_revision', 'research_update',
                'native_condition_research', 'refit'}):
            self._refresh_parent_research_phase(owner)
            return
        execution = {'action': action, 'model_call': self.costs['model_calls'],
                     **{key: deepcopy(arguments[key]) for key in
                        ('operator', 'model', 'component', 'view', 'target') if key in arguments},
                     **{key: deepcopy(result[key]) for key in
                        ('returncode', 'mse', 'status', 'error', 'model_dependencies', 'research_bindings',
                         'representation_diagnostic', 'training_representation_diagnostic')
                        if key in result}}
        focus['executions'].append(execution)
        if operator == 'refit':
            validation, split = result.get('validation', {}), result.get('split', {})
            review = {
                **{key: deepcopy(result[key]) for key in
                   ('status', 'source_model', 'tested_model', 'error', 'paid_rows_at_refit')
                   if key in result},
                'validation_mse': validation.get('mse'),
                'validation_returncode': validation.get('returncode'),
                **{key: deepcopy(split[key]) for key in
                   ('train_rows', 'holdout_rows', 'validation_records') if key in split},
            }
            execution.update(deepcopy(review), evaluation_scope='paid_trajectory_holdout')
            focus['method_review_result'] = review
            # The existing candidate record holds this source's trajectory
            # validation; the exported context subsequently fits selected rows.
            focus['evaluated_model'] = arguments['model']
            self._refresh_parent_research_phase(owner)
        elif operator == 'research_update':
            relation = result.get('relation_research', {})
            execution.update(question_id=result.get('research_question_id'),
                             structural_status=relation.get('status'),
                             structural_diagnosis=relation.get('diagnosis'),
                             exported_models=deepcopy(relation.get('exported_models', [])))
            self._refresh_parent_research_phase(owner)
        elif operator == 'native_condition_research':
            research = result.get('native_condition_research', {})
            execution.update(question_id=result.get('research_question_id'),
                             structural_status=research.get('status'),
                             exported_models=deepcopy(research.get('exported_models', [])))
            self._refresh_parent_research_phase(owner)
        if (action in {'evaluate', 'finish'} and 'component' not in arguments
                and arguments.get('target', 'response') == 'response'):
            model = arguments.get('model', 'model.py')
            path = self.workspace / model
            source = path.read_text() if path.is_file() else None
            changed = (source is not None and focus['baseline_parent_source'] is not None
                       and source != focus['baseline_parent_source'])
            used = [entry for entry in result.get('model_dependencies', []) if entry.get('calls', 0) > 0]
            inputs = result.get('research_bindings', [])
            execution.update(source_changed=changed, consumed_components=deepcopy(used),
                             consumed_parameters=deepcopy(inputs))
            focus['evaluated_model'] = model
            self._refresh_parent_research_phase(owner)
            if result.get('fresh_confirmation', {}).get('passed') is False:
                focus['phase'] = 'candidate_revision'
        elif operator in {'analyze_evidence', 'analyze_revision'}:
            answer = result.get('revision', {}).get('returned_answer', {})
            question_id = result.get('research_question_id')
            if (question_id in {owner['id'], focus['parent_id']}
                    and answer.get('evidence_status') == 'refuted'
                    and answer.get('check_results')):
                execution['checked_answer'] = deepcopy(answer)
                question = next(q for q in self.questions if q['id'] == question_id)
                same_assignment = all(question.get(key) == question['assignment'].get(key)
                                      for key in ('question', 'parent_id', 'depends_on'))
                # A narrower local hypothesis can be refuted while its original
                # parent investigation still needs a revised executable answer.
                if same_assignment:
                    focus.update(status='refuted', completed_at_model_call=self.costs['model_calls'])
                else:
                    self._refresh_parent_research_phase(owner)

    def _finish_local_research(self, arguments, laboratory):
        """Score explicitly returned predictors through the existing paid evaluator."""
        result = {'status': 'inquiry_finished', 'lesson': arguments.get('lesson', '')}
        if 'model' in arguments:
            evaluation = self._execute('evaluate', {key: deepcopy(value) for key, value in arguments.items()
                if key not in {'lesson', 'artifacts', 'inconclusive'}}, laboratory)
            if evaluation.get('returncode') != 0 or evaluation.get('error'):
                return {**evaluation, 'status': 'inquiry_delivery_failed'}
            result['evaluation'] = evaluation
        return result

    def _repair_repeated_execution_error(self):
        """Return a repeated Python failure to the existing local repair loop."""
        from .temporal_paired_method import _failure_state

        if (self.provider is None or self._inquiry_stack or self.model is not None
                or self.config.max_model_calls - self.costs['model_calls'] <= 1):
            return
        recent = [(index, event) for index, event in enumerate(self.events)
                  if event.get('action') == 'run' and not event.get('arguments', {}).get('operator')]
        if len(recent) < 2 or recent[-1][0] != len(self.events) - 1:
            return
        (previous_index, previous), (index, event) = recent[-2:]
        feedback = event.get('feedback', {})
        path = feedback.get('executed_path')
        if (not path or feedback.get('execution_repair') or feedback.get('timeout')
                or not (feedback.get('returncode') or feedback.get('error'))):
            return
        failure = _failure_state(feedback)
        if failure != _failure_state(previous.get('feedback', {})):
            return
        parent_id = feedback.get('research_question_id') or self._active_question_id
        previous_id = previous.get('feedback', {}).get('research_question_id') or parent_id
        if previous_id != parent_id:
            return
        parent = next((q for q in self.questions if q['id'] == parent_id), None)
        question = {
            'id': f'execution_repair_{index}', 'parent_id': parent_id, 'depends_on': [],
            'question': f'Repair the repeated execution error in {path}, preserving the '
                'author\'s existing calculation and research objective. Inspect the source and '
                'actual error, edit it yourself, and execute the repaired calculation. '
                'Use local finish to return the artifact and execution outcome to the parent; '
                'successful execution is not scientific confirmation.',
            'origin': 'host_execution_repair', 'failed_program_path': path,
            'failure_event_indices': [previous_index, index], 'execution_failure': failure,
            'parent_assignment': deepcopy(parent.get('assignment', parent)) if parent else None,
            'parent_question': parent.get('question') if parent else None,
        }
        feedback['execution_repair'] = self._start_inquiry(
            {'question': question, 'max_model_calls': 4}, parent_event_index=index)

    def _procedure_terminal_status(self):
        if self._inquiry_stack and self._inquiry_stack[-1]['awaiting_first_turn']:
            return 'inquiry_started'
        if self._inquiry_stack and self._inquiry_stack[-1]['stop_requested']:
            return 'inquiry_finished'
        return super()._procedure_terminal_status()

    def _open_question(self, question, experiment_path=None):
        """Reuse one live question; acquisition history is not a new assignment."""
        if not isinstance(question, dict):
            raise ValueError('research_question must be a JSON object')
        old = next((q for q in self.questions if q['id'] == question['id']), None)
        version = old.get('meaning_version', 1) if old else 1
        current = open_assigned_question(self, question, experiment_path)
        if current.get('meaning_version', 1) != version:
            for revision in self.revisions:
                if revision.get('question_id') == question['id']:
                    revision.setdefault('question_meaning_version', version)
        if current.get('meaning_version', 1) != version and question['id'] in self.relation_research:
            # A historical bank remains recallable, but its support answers the
            # old meaning. Keep its original position and all exported models.
            key = f"{question['id']}@meaning{version}"
            self.relation_research = {(key if name == question['id'] else name): value
                                     for name, value in self.relation_research.items()}
        return current

    def _return_revision_answer(self, question, record, answer):
        record_answer(self, question, answer)
        parent_id = question.get('parent_id')
        if self._inquiry_stack and self._inquiry_stack[-1]['question_id'] == question['id']:
            parent_id = self._inquiry_stack[-1]['parent_question']
        parent = next((q for q in self.questions if q["id"] == parent_id), None)
        if parent is not None:
            parent.setdefault("child_answers", []).append({
                "question_id": question["id"], "revision_index": record["index"], **deepcopy(answer)})
            parent["status"] = "reopened"
            if self.arm == "hegel":
                self._active_question_id = parent["id"]

    def _return_structural_answer(self, question, answer):
        """Return a measured local relation without inventing a purchased revision."""
        if question.get('answer') is not None:
            question.setdefault('structural_answer_history', []).append(deepcopy(question['answer']))
        question.update(status=answer['status'], answer=deepcopy(answer['answer']),
                        evidence=deepcopy(answer['evidence']))
        record_answer(self, question, answer)
        parent_id = question.get('parent_id')
        if self._inquiry_stack and self._inquiry_stack[-1]['question_id'] == question['id']:
            parent_id = self._inquiry_stack[-1]['parent_question']
        parent = next((q for q in self.questions if q['id'] == parent_id), None)
        if parent is not None:
            parent.setdefault('child_answers', []).append({
                'question_id': question['id'], **deepcopy(answer)})
            parent['status'] = 'reopened'

    def _prepare_research_model(self, model):
        return before_candidate(self, model)

    def _query(self, arguments, laboratory):
        reviewed = {q["id"]: deepcopy(q.get("child_answers", [])) for q in self.questions}
        result = super()._query(arguments, laboratory)
        returned = result.get("question")
        if returned is not None:
            question = next(q for q in self.questions if q["id"] == returned["id"])
            self._active_question_id = question["id"]
            question["reviewed_child_answers"] = reviewed.get(question["id"], [])
            acquired = [index for observation in result.get("observations", [])
                        for index in observation.get("row_indices", [observation["observation_index"]])]
            question["acquired_evidence"] = sorted(set(question.get("acquired_evidence", [])) | set(acquired))
            # The inherited operator executor used append-only questions. A
            # continued question still owns its actual continuation feedback.
            for key in ("after_observation_results", "after_observation_error"):
                if key in result:
                    question[key] = deepcopy(result[key])
            revision = result.get("revision")
            if revision is not None:
                self.revisions[revision["index"]]["question_id"] = question["id"]
                result["revision"] = deepcopy(self.revisions[revision["index"]])
            if acquired:
                prepare_evidence_readout(self, question)
            # The canonical question owns every paid native research round.
            # Re-embedding that cumulative bank in each query feedback makes
            # method traces grow quadratically, then repeats it in the author.
            result["question"] = {key: deepcopy(value) for key, value in question.items()
                                  if key != "native_condition_research"}
            if "native_condition_research" in question:
                result["question"]["native_condition_research"] = {
                    "full_content_file": "research_state.json",
                    "json_path": ["questions", self.questions.index(question),
                                  "native_condition_research"],
                    "round_count": len(question["native_condition_research"])}
        reassessments = refresh_observables(self, laboratory)
        if reassessments:
            result["observable_reassessments"] = reassessments
        component_reassessments = refresh_components(self, laboratory)
        if component_reassessments:
            result["component_reassessments"] = component_reassessments
        return result

    def _execute(self, action, arguments, laboratory):
        arguments = deepcopy(arguments)
        pending_parent = arguments.pop('pending_question', None) if action == 'query' else None
        if pending_parent is not None:
            arguments = pending_query_arguments(self, pending_parent)
        parent_research = self._parent_research()
        if any(frame['question'].get('origin') == 'host_execution_repair'
               for frame in self._inquiry_stack):
            if (action not in {'write', 'run', 'finish'}
                    or (action == 'run' and arguments.get('operator') is not None)):
                raise ValueError('Execution repair uses ordinary write/run and local finish; '
                                 'return scientific actions and acquisitions to the parent.')
        if "notes" in arguments:
            self._save("research_notes.md", arguments.pop("notes"))
        if action == 'run' and arguments.get('operator') == 'inquire':
            return self._start_inquiry(arguments)
        if action == 'finish' and self._inquiry_stack:
            result = self._finish_local_research(arguments, laboratory)
            if result['status'] == 'inquiry_finished':
                self._inquiry_stack[-1]['stop_requested'] = True
            return result
        if action == 'finish' and parent_research is not None and arguments.get('inconclusive') is True:
            focus = parent_research['parent_research']
            focus.update(status='inconclusive', lesson=arguments.get('lesson', ''),
                         completed_at_model_call=self.costs['model_calls'])
            return {'status': 'parent_research_inconclusive', 'lesson': focus['lesson'],
                    'research_question_id': focus['parent_id']}
        native_arguments = (deepcopy(arguments)
            if action == "run" and arguments.get("operator") == "native_condition_research" else None)
        declaration = arguments.pop("research_question", None) or arguments.get("question")
        if declaration is None and self._inquiry_stack:
            declaration = {'id': self._inquiry_stack[-1]['question_id']}
        elif declaration is None and parent_research is not None:
            focus = parent_research['parent_research']
            conflict = next((q for q in self.questions
                if q['id'] == self._active_question_id
                and q['id'] == f"representation:{focus.get('evaluated_model')}"
                and q.get('origin') == 'computed_representation_conflict'
                and q.get('status') == 'open'), None)
            declaration = {'id': conflict['id'] if conflict else focus['parent_id']}
        question = self._open_question(declaration) if declaration else None
        reviewed_returns = deepcopy(question.get("child_answers", [])) if question is not None else []
        if question is not None:
            self._active_question_id = question["id"]
            if action == "query":
                arguments.setdefault("question", {key: question[key]
                    for key in ("id", "question", "parent_id", "depends_on")})
        written = {}
        if "code" in arguments and (action == "evaluate" or (
                action == "run" and arguments.get("operator") == "refit")):
            model = arguments.get("model", "model.py")
            written = super()._execute("write", {"path": model, "code": arguments.pop("code")}, laboratory)
        if action in {"evaluate", "finish"} and "bindings" in arguments:
            path = self.workspace / "model_bindings.json"
            bindings = json.loads(path.read_text()) if path.exists() else {}
            bindings[arguments.get("model", "model.py")] = arguments.pop("bindings")
            self._save("model_bindings.json", json.dumps(bindings, indent=2))
            bind_components(self, arguments.get("model", "model.py"),
                            bindings[arguments.get("model", "model.py")])
        rebuilding = None
        candidate_action = action in {"evaluate", "finish"} or (
            action == "run" and arguments.get("operator") == "refit")
        claim = arguments.pop('claim', None) if candidate_action else None
        if candidate_action:
            model = arguments.get("model", "model.py")
            if "research_inputs" in arguments:
                rebuilding = materialize(self, model, arguments.pop("research_inputs"))
            else:
                try:
                    rebuilding = before_candidate(self, model)
                except ValueError as error:
                    return {"error": str(error), "status": "dependency_unresolved",
                            "research_repairs": research_repairs(self)}
        if action == 'run' and arguments.get('operator') == 'theory_consolidation':
            result = {'operator': 'theory_consolidation', **consolidate_research(self, arguments, laboratory, question)}
            self._prepare_program_state()
        elif action == 'run' and arguments.get('operator') == 'theory_transfer':
            result = {'operator': 'theory_transfer', **transfer_theory(self, arguments, laboratory)}
            self._prepare_program_state()
        elif action == 'run' and arguments.get('operator') == 'research_route':
            result = {'operator': 'research_route', **route_research(self, arguments, laboratory, question)}
            revision = result.get('research_track_question_ref', result.get('parent_revision_question_ref'))
            if revision is not None:
                from .finite_native_author import export_revision_candidates
                from .research_episodes import _question_versions
                owner = _question_versions(self)[revision['question_id'], revision.get('meaning_version', 1)][0]
                result['parent_revision_exports'] = export_revision_candidates(self, owner)
            self._prepare_program_state()
        elif action == 'run' and arguments.get('operator') == 'research_step':
            if question is None:
                raise ValueError('research_step requires a named research_question')
            result = {'operator': 'research_step', **step_question(self, question, laboratory)}
            self._prepare_program_state()
        elif action == "evaluate" and "component" in arguments:
            result = evaluate_component(self, arguments, laboratory, question)
        elif action == "evaluate" and arguments.get("target", "response") != "response":
            result = evaluate_observable(self, arguments, laboratory, question)
        elif action == "run" and arguments.get("operator") == "compare_experiments":
            result = {"operator": "compare_experiments", "comparison": compare_experiments(
                self, arguments.get("models"), arguments.get("experiments"), laboratory,
                mse_threshold=self.config.confirmation_mse_threshold, target=arguments.get("target", "response"),
                reference_actions=arguments.get("reference_actions"))}
        elif action == "run" and arguments.get("operator") == "research_update":
            if question is None:
                raise ValueError('research_update requires an authored research question')
            result = execute_structural_research(self, arguments, laboratory, question)
        elif action == "run" and arguments.get("operator") == "native_condition_research":
            before_compute = self.costs["compute_cost"]
            before_seconds = self.costs["python_execution_seconds"]
            try:
                result = execute_native_condition_research(self, arguments, laboratory, question)
            except (ValueError, TypeError, KeyError, OSError) as error:
                result = {"operator": arguments["operator"], "error": str(error),
                    "compute_cost": self.costs["compute_cost"] - before_compute,
                    "python_execution_seconds": self.costs["python_execution_seconds"] - before_seconds}
                self.operator_results.append({"index": len(self.operator_results),
                    "arguments": native_arguments, **result})
                self._prepare_program_state()
                raise
            if question is not None:
                result['question_meaning_version'] = question.get('meaning_version', 1)
            self.operator_results.append({"index": len(self.operator_results),
                "arguments": native_arguments, **deepcopy(result)})
            self._prepare_program_state()
        elif action == "run" and arguments.get("operator") == "analyze_revision":
            index = arguments["revision_index"]
            if type(index) is not int or not 0 <= index < len(self.revisions):
                raise ValueError("revision_index must identify an existing experiment")
            record = self.revisions[index]
            owner = next((q for q in self.questions if q["id"] == record.get("question_id")), None)
            result = {"operator": "analyze_revision", "revision": analyze_revision(
                self, record, laboratory, owner, path=arguments.get("path"))}
            if owner is not None:
                result["question"] = deepcopy(owner)
        elif action == "run" and arguments.get("operator") == "analyze_evidence":
            result = {"operator": "analyze_evidence", "revision": analyze_paid_evidence(
                self, arguments["path"], laboratory, question,
                target=arguments.get("target", "response"),
                record_indices=arguments.get("record_indices"))}
            if question is not None:
                result["question"] = deepcopy(question)
        elif action == "run" and arguments.get("operator") == "bind_research":
            result = {"operator": "bind_research", **materialize(
                self, arguments.get("model", "model.py"), arguments.get("inputs"))}
        else:
            result = super()._execute(action, arguments, laboratory)
        result = {**written, **result}
        structural_trial = trial_structural_proposal(self, action, arguments, laboratory)
        if structural_trial is not None:
            result['structural_trial'] = structural_trial
        readout_trial = trial_evidence_readout(self, action, arguments, laboratory)
        if readout_trial is not None:
            result['evidence_readout_trial'] = readout_trial
        if action == "write" and arguments.get("path") in self.research_candidates:
            candidate = self.research_candidates[arguments["path"]]
            candidate['status'] = 'needs_refit'
            candidate.pop('current_evaluation', None)
        if rebuilding is not None:
            result["research_input_update"] = rebuilding
        if candidate_action:
            evaluation = result
            if action == "run" and arguments.get("operator") == "refit":
                evaluation = {**result.get("validation", result),
                              "evaluation_scope": "paid_trajectory_holdout"}
            record_evaluation(self, arguments.get("model", "model.py"), evaluation, claim=claim)
        if research_repairs(self):
            result["research_repairs"] = research_repairs(self)
        if (action in {"evaluate", "finish"} and "component" not in arguments
                and arguments.get("target", "response") == "response"):
            self.model_dependencies = deepcopy(result.get("model_dependencies", []))
        if question is not None:
            # This records the decision's actual question association only.
            # Answers produced by this action remain new to a later parent turn.
            question["reviewed_child_answers"] = reviewed_returns
            result["research_question_id"] = question["id"]
            result['question_meaning_version'] = question.get('meaning_version', 1)
            if action == 'run' and arguments.get('operator') == 'research_choice':
                from .research_episodes import _question_versions
                evaluated_paths = {item.get('path') for item in result.get('evaluations', [])}
                owners = [owner for owner, _ in _question_versions(self).values()
                          if any(item.get('model') in evaluated_paths
                                 for item in owner.get('revision_candidate_exports', []))]
                combined = None
                for owner in owners or [question]:
                    recorded = record_revision_comparison_failures(
                        self, owner, result, source_ref={'file': 'research_state.json',
                            'json_path': ['operator_results', len(self.operator_results) - 1]},
                        comparison_question=question)
                    if combined is None:
                        combined = deepcopy(recorded)
                        combined['owner_results'] = []
                    else:
                        for key, value in recorded.items():
                            if type(value) is int:
                                combined[key] += value
                            elif isinstance(value, list):
                                combined[key].extend(deepcopy(value))
                    combined['owner_results'].append({
                        'question_ref': {'question_id': owner['id'],
                            'meaning_version': owner.get('meaning_version', 1)},
                        'new_failure_record_count': recorded.get('new_failure_record_count', 0),
                        'unknown_count': len(recorded.get('unknown', []))})
                if combined.get('unknown'):
                    combined['status'] = 'UNKNOWN'
                result['revision_comparison_failures'] = combined
        self._record_parent_research(parent_research, action, arguments, result)
        if pending_parent is not None and result.get('observations'):
            parent = next(q for q in self.questions if q['id'] == pending_parent)
            executed = {
                'query': parent.pop('pending_query'),
                'source': parent.pop('pending_query_source'),
                'record_indices': deepcopy(result.get('record_indices', []))}
            parent.setdefault('executed_pending_queries', []).append(executed)
            conditional = continue_conditional_plan(self, parent, executed, result)
            if conditional is not None:
                result['conditional_continuation'] = conditional
        return result

    def _manifest_error(self, manifest, model):
        if not self.source_task:
            # Continuation can finish ordinary scientific research without
            # inventing a select_actions program merely to populate a list.
            # The inherited consumer records the selected executable model;
            # paid and fresh scoring, not this bookkeeping, determine adoption.
            return None
        return super()._manifest_error(manifest, model)

    def _executed_python_analyses(self):
        return [(index, event) for index, event in enumerate(self.events)
                if event.get('action') == 'run' and not event.get('arguments', {}).get('operator')
                and ('returncode' in event.get('feedback', {})
                     or any(event.get('feedback', {}).get(key)
                            for key in ('stdout', 'stderr', 'error', 'timeout'))
                     or event.get('feedback', {}).get('status') == 'runtime_interrupted')]

    def _online_research(self):
        pending = []
        for record in self.revisions:
            if (record.get("observations") or record.get("kind") == "paid_evidence_analysis") and (record.get("readout_error") or not record.get("analysis_is_agent_authored")):
                pending.append({"revision_index": record["index"], "question_id": record.get("question_id"),
                    "status": record["status"], "record_indices": record["record_indices"],
                    "readout": record["proposal"].get("readout"),
                    "error": record.get("readout_error"),
                    "record_file": "research_state.json"})
        returns = []
        for question in self.questions:
            reviewed = question.get("reviewed_child_answers", [])
            returned = [answer for answer in question.get("child_answers", []) if answer not in reviewed]
            if returned:
                returns.append({"parent_id": question["id"], "answers": deepcopy(returned)})
        questions = [{key: deepcopy(q[key]) for key in
            ("id", "question", "assignment", "parent_id", "depends_on", "status", "experiment_path",
             "answer", "child_answers", "evidence", "acquired_evidence", "revision_index",
             "executable_answer", "evidence_status", "check_results", "representation_diagnostic", "analysis_scope",
             "evidence_readout_proposal", "meaning_version", "kind", "context_ref", "query_semantics",
             "answer_type", "answer_resolution_ref", "use_sites", "compilation", "semantic_answer",
             "support_path_status", "pending_author_work", "pending_query", "pending_query_source",
             "queue", "origin", "agenda_basis")
            if key in q} for q in self.questions]
        by_id = {question['id']: question for question in questions}
        for summary, question in zip(questions, self.questions):
            if question.get('routing_execution'):
                execution = question['routing_execution']
                summary['research_routing'] = {'version': execution['version'],
                    'status': execution['result']['status'],
                    'source': {'file': 'research_state.json',
                               'question_id': question['id'], 'field': 'routing_execution'}}
        for index, event in self._executed_python_analyses():
            feedback = event['feedback']
            question = by_id.get(feedback.get('research_question_id'))
            if question is not None:
                question.setdefault('executed_analyses', []).append({
                    'ref': f'analysis:{index}', 'event_index': index,
                    **{key: deepcopy(feedback[key]) for key in
                       ('executed_path', 'returncode', 'timeout', 'error', 'status') if key in feedback},
                    'source': {'file': 'research_context.json',
                               'json_path': ['events', index, 'feedback']}})
        reviewed_evidence = {}
        for record in self.revisions:
            cited = reviewed_evidence.setdefault(record.get("question_id"), set())
            answers = [record.get("returned_answer")]
            answers.extend(attempt.get("authored_answer")
                           for attempt in record.get("analysis_attempts", []) if not attempt.get("error"))
            for answer in answers:
                if isinstance(answer, dict):
                    cited.update(answer.get("evidence", []))
        for summary, question in zip(questions, self.questions):
            summary["unreviewed_evidence"] = [index for index in question.get("acquired_evidence", [])
                if index not in reviewed_evidence.get(question["id"], set())]
        latest = self.events[-1] if self.events else {}
        feedback = latest.get("feedback", {})
        comparison = self.experiment_comparisons[-1] if self.experiment_comparisons else None
        latest_comparison = ({key: deepcopy(comparison[key]) for key in
            ("index", "status", "diagnosis", "target", "evidence_cutoff", "models", "eligible_models",
             "mse_threshold", "selected_experiment", "query_arguments", "compute_cost", "error",
             "selection_policy", "task_value")
            if key in comparison} if comparison is not None else None)
        if latest_comparison is not None:
            latest_comparison["prior_fit"] = [{key: deepcopy(fit[key]) for key in
                ("model", "returncode", "mse", "error") if key in fit}
                for fit in comparison["prior_fit"]]
            latest_comparison["comparisons"] = [{key: deepcopy(experiment[key]) for key in
                ("id", "row_count", "observation_cost", "predicted_separation")}
                for experiment in comparison["comparisons"]]
            latest_comparison["current_paid_rows"] = len(self.rows)
        paid_records = []
        for index in range(max(0, len(self.observations) - 4), len(self.observations)):
            record = self.observations[index]
            rows = record.get("row_indices", [])
            paid_records.append({"record_index": index, "row_count": len(rows),
                "response_row_range": [min(rows), max(rows)] if rows else [],
                "action": deepcopy(record.get("action"))})
        review = next((outcome for outcome in reversed(self.method_results)
                       if outcome["feedback"].get("status") == "method_review_required"), None)
        if (review is not None and review['feedback']['review'].get('kind') == 'research_evidence_extended'
                and native_evidence_extension_review(self, review['path']) is None):
            review = None
        return {"active_question": self._active_question_id, "questions": questions,
                "pending_analyses": pending, "parent_returns": returns,
                "current_paid_rows": len(self.rows), "paid_record_count": len(self.observations),
                "recent_paid_records": paid_records,
                "recent_revisions": [revision_evidence_scope(self, record) for record in self.revisions[-4:]],
                "latest_method_review": ({**deepcopy(review["feedback"]["review"]),
                    "method_step": review["step"],
                    "subsequent_method_steps": len(self.method_results) - review["step"] - 1}
                    if review is not None else None),
                "latest_comparison": latest_comparison,
                "latest_execution": {"action": latest.get("action"), **{key: deepcopy(feedback[key])
                    for key in ("error", "status", "executed_path", "mse", "returncode") if key in feedback}},
                "evidence_file": "research_state.json"}

    def _shared_files(self, *, excluded=()):
        """Read authored work without serializing immutable fitting snapshots."""
        files = {}
        for directory, subdirs, names in os.walk(self.workspace / "shared"):
            subdirs[:] = [name for name in subdirs if name != "_evaluated_model"]
            for name in names:
                path = Path(directory) / name
                relative = str(path.relative_to(self.workspace))
                if relative not in excluded and path.suffix in {".py", ".json", ".md", ".txt"}:
                    files[relative] = path.read_text(encoding="utf-8")
        return files

    def _request(self, laboratory):
        for name in list(self.files):
            if "_evaluated_model" in Path(name).parts:
                self.files.pop(name)
                continue
            path = self.workspace / name
            if path.is_file():
                self.files[name] = path.read_text(encoding="utf-8")
        self.files.update(self._shared_files())
        request = super()._request(laboratory)
        request["research_phase"] = "source_learning" if self.source_task else "continuation"
        if not self.source_task:
            # Restore the existing ProcedureConsumer contract after replacing
            # its instrument-specific interface with the common archive one.
            request["task_interface"] = request["task_interface"].replace(
                _SOURCE_EXECUTION_REQUIREMENT, _CONTINUATION_REQUIREMENT)
            start = COMMON_INTERFACE.index("For the hegel condition, retain")
            end = COMMON_INTERFACE.index("During source learning, Hegel")
            request["task_interface"] = request["task_interface"].replace(
                COMMON_INTERFACE[start:end],
                "Executable hypotheses, paid evidence and research files persist. During "
                "continuation, no manual manifest or new experimental program is needed "
                "to submit a predictor for fresh confirmation.\n")
        notes = self.workspace / "research_notes.md"
        request["research_notes"] = notes.read_text(encoding="utf-8") if notes.exists() else ""
        request["online_research"] = self._online_research()
        request["evidence_notebook"] = self._evidence_notebook()
        request["observable_models"] = deepcopy(self.observable_models)
        request["conditional_models"] = deepcopy(self.conditional_models)
        request["model_dependencies"] = deepcopy(self.model_dependencies)
        request["research_repairs"] = research_repairs(self)
        request["research_input_bindings"] = deepcopy(self.research_input_bindings)
        request['relation_research'] = {identifier: structural_summary(entry)
            for identifier, entry in self.relation_research.items()}
        parent_research = self._parent_research()
        if parent_research is not None:
            focus = parent_research['parent_research']
            request['parent_research_focus'] = {
                'question_id': parent_research['id'], 'parent_id': focus['parent_id'],
                'origin': 'host_parent_research_continuation',
                'status': focus['status'], 'assignment': deepcopy(parent_research['assignment']),
                'returned_research': deepcopy(focus.get('returned_research')
                                               or parent_research['review_return']),
                'baseline_parent': deepcopy(focus['baseline_parent']),
                'executions': deepcopy(focus['executions']),
                **{key: deepcopy(focus[key]) for key in
                   ('phase', 'evaluated_model', 'evaluation_current', 'candidate_paid_rows', 'current_paid_rows',
                    'method_review_result')
                   if key in focus},
            }
            request['task_interface'] = (
                'Continue the parent investigation from parent_research_focus. Turn the returned '
                'local work into a tested executable result, or explicitly reject or leave it '
                'unresolved using its evidence. Choose the relation, applicability, representation '
                'and program yourself. For a local relation, write its observation view and '
                'predictor and use evaluate {component,view,model}; later purchases re-evaluate '
                'that frozen component on applicable new evidence. Compose chosen components '
                'through bindings/bind_models, or write a revised whole parent predictor, then '
                'evaluate it on the paid evidence. An ordinary calculation, note, file write or '
                'unchanged baseline evaluation does not complete this handoff. High paid error '
                'keeps the parent under revision; source changes and executed dependencies alone '
                'do not resolve it. If phase is prediction_method_revision, revisit the actual '
                'trajectory-validation result and repeated frozen prediction failures. Another '
                'low training error does not discharge that research obligation; choose a revised '
                'explanation or fitting procedure, a distinguishing experiment, or an unresolved '
                'return. Normal fresh confirmation remains available, including predictors without '
                'a callable fitting procedure. If phase is representation_unassessed, expose your actual '
                'downstream inputs through the existing represent/predict_features contract. '
                'If phase is representation_conflict, investigate the actual conflicting rows '
                'before merely refitting the same inputs. A current paid fit meeting the threshold '
                'with no known representation conflict only awaits independent confirmation; '
                'new paid evidence requires reassessment. Only a passed fresh confirmation '
                'completes the parent successfully. You may instead return '
                'finish {inconclusive:true,lesson:"remaining obstacle"}; that ends this handoff '
                'without submitting a model or buying confirmation. Normal parent finish retains '
                'the existing independent-confirmation contract.\n\n' + request['task_interface'])
        assignment = self._scientific_revision_assignment()
        if assignment is not None:
            request['scientific_revision_assignment'] = assignment
            review = next(q for q in self.questions if q['id'] == assignment['id'])
            if review.get('review_return') and not self._inquiry_stack:
                request['scientific_revision_review'] = {
                    'phase': 'returned_to_parent', **deepcopy(review['review_return'])}
                request['task_interface'] = (
                    'The designated failed-confirmation review has returned to the parent. '
                    'Use its actual executable results and evidence to decide whether to adopt, '
                    'revise or reject its work. Parent finish now requests ordinary independent '
                    'confirmation; the review return itself is not scientific confirmation.\n\n'
                    + request['task_interface'])
        if self.arm == "hegel":
            root = next(q for q in self.questions if q["id"] == "task")
            request["research_root"] = deepcopy(root["assignment"])
            request["research_dependency_focus"] = dependency_focus(self)
        if self._inquiry_stack:
            frame = self._inquiry_stack[-1]
            session = self.llm_inquiries[frame['question_id']]
            request['enclosing_research_root'] = request.get('research_root')
            request['research_root'] = {key: deepcopy(frame['assignment'].get(key))
                                        for key in ('id', 'question', 'parent_id')}
            request['research_notes'] = session['notes']
            request['llm_inquiry'] = {
                'question_id': frame['question_id'], 'depth': len(self._inquiry_stack),
                'assignment': deepcopy(frame['assignment']), 'question': deepcopy(frame['question']),
                'remaining_model_calls': frame['call_limit'] - self.costs['model_calls'],
                'invocation': session['invocations'], 'event_indices': list(frame['event_indices'])}
            request['task_interface'] = (
                'Resolve llm_inquiry.assignment using the current paid evidence and tools, then '
                'return your executable work and its evidence to the caller. The apparatus task '
                'below is the enclosing prediction task. Local finish only returns to the parent; '
                'it neither requests response confirmation nor proves the child hypothesis. '
                'An explicitly returned model in local finish is evaluated on paid evidence first; '
                'include component and view for an authored local relation. Execution failure '
                'returns to this same researcher for repair.\n\n'
                + request['task_interface'])
            if frame['question'].get('origin') == 'host_execution_repair':
                request['task_interface'] = (
                    'This is a local execution repair, not scientific hypothesis research. '
                    'llm_inquiry.assignment names the original source, parent objective, and '
                    'actual repeated failure. Only ordinary write/run and local finish are '
                    'available here; use existing files and paid records. Preserve the author\'s '
                    'calculation goal, execute your edit, and return its actual outcome. '
                    'Leave new observations, model selection, and scientific conclusions to '
                    'the parent.\n\n' + request['task_interface'])
            proposal = structural_proposal_context(self)
            if proposal is not None:
                request['structural_proposal'] = proposal
                request['task_interface'] = STRUCTURAL_PROPOSAL_INSTRUCTION + request['task_interface']
        readout = evidence_readout_context(self)
        if readout is not None:
            request['evidence_readout_proposal'] = readout
            request['task_interface'] = EVIDENCE_READOUT_INSTRUCTION + request['task_interface']
        return request

    def _evidence_notebook(self):
        """Render empirical memory from executed records, separately from authored prose."""
        notes = self.workspace / 'research_notes.md'
        note_text = notes.read_text(encoding='utf-8') if notes.exists() else ''

        def cited_ranges(kind):
            return [(int(start), int(end or start)) for start, end in re.findall(
                rf'\b{kind}:(\d+)(?:\s*[-–]\s*(\d+))?', note_text)]

        def revision_fact(record):
            fact = {
                **revision_evidence_scope(self, record),
                'status': record['status'],
                'analysis_version': len(record.get('analysis_attempts', [])),
                'authored_readout_status': (record.get('returned_answer') or {}).get('status'),
                'evidence_status': record.get('evidence_status'),
                'check_scope': 'Numerical checks evaluate authored weighted sums of paid labels. '
                    'They do not validate experiment-role descriptions or the scientific interpretation. '
                    'record_references contains the actual complete public inputs.',
                'source': {'file': 'research_state.json', 'json_path': ['revisions', record['index']]}}
            path = (record.get('proposal') or {}).get('readout')
            if path:
                arguments = {'operator': 'analyze_revision', 'revision_index': record['index'],
                             'path': path}
                if record.get('question_id'):
                    arguments['research_question'] = {'id': record['question_id']}
                fact['reanalysis_action'] = {'action': 'run', 'arguments': arguments}
                fact['reanalysis_source'] = (
                    'Runs the current authored file at arguments.path, which may have been reused '
                    'for a later analysis. The old source is not restored. Inspect the saved source '
                    'and edit that file or select another path before reanalyzing the fixed evidence.')
            attempts = record.get('analysis_attempts', [])
            if attempts and 'source' in attempts[-1]:
                fact['executed_readout_source'] = {'file': 'research_state.json', 'json_path': [
                    'revisions', record['index'], 'analysis_attempts', len(attempts) - 1, 'source']}
            return fact

        checked_relations = []
        for index in range(len(self.revisions)):
            try:
                relations = _local_relations(self, {'revision_index': index})
            except ValueError:
                continue
            checks = relations['checks']
            checked_relations.append({
                'revision_index': index, 'version': relations['version'],
                'question_id': relations['question_id'], 'check_count': len(checks),
                'satisfied_count': sum(check['status'] == 'satisfied' for check in checks),
                'violated_count': sum(check['status'] == 'violated' for check in checks),
                'selector': {'revision_index': index},
                'source': {'file': 'research_state.json', 'json_path': ['revisions', index]}})
        facts = {'response_data': {
            'kind': 'data_contract', 'file': 'observations.json', 'rows': len(self.rows),
            'row_fields': ['features', 'target', 'history'],
            'label': 'One scalar target for current features after the preceding control history.',
            'history': 'Ordered preceding controls since this experiment reset; earlier targets '
                       'belong to their own response rows.',
            'indexing': 'Row identifiers are positions in the observations.json array. '
                        'laboratory_observations.json records use row_indices to group those '
                        'positions into experiments.'}}
        selected = {record['index'] for record in self.revisions[-4:]}
        selected.update(q['revision_index'] for q in self.questions
                        if q['id'] == self._active_question_id and 'revision_index' in q)
        cited_revisions = cited_ranges('revision')
        selected.update(record['index'] for record in self.revisions
                        if any(start <= record['index'] <= end for start, end in cited_revisions))
        for record in self.revisions:
            if record['index'] not in selected:
                continue
            facts[f"revision:{record['index']}"] = revision_fact(record)
        for index in range(max(0, len(self.observations) - 4), len(self.observations)):
            record = self.observations[index]
            rows = record.get('row_indices', [])
            facts[f'record:{index}'] = {
                'kind': 'paid_observation', 'record_index': index,
                'response_row_range': [min(rows), max(rows)] if rows else [],
                'response_row_count': len(rows),
                'forecast_revision_indices': [r['index'] for r in self.revisions
                    if index in r.get('record_indices', []) and r.get('forecasts')],
                'source': {'file': 'laboratory_observations.json', 'json_path': [index]}}
        for index in range(max(0, len(self.confirmations) - 2), len(self.confirmations)):
            confirmation = self.confirmations[index]
            facts[f'confirmation:{index}'] = {'kind': 'independent_confirmation',
                **{key: deepcopy(confirmation[key]) for key in
                   ('passed', 'mse', 'rows', 'mse_threshold', 'observation_indices',
                    'returncode', 'timeout', 'model', 'submitted_candidate', 'residuals') if key in confirmation},
                'confirmation_index': index}
        evaluated = self._evaluated_summary()
        if evaluated:
            facts['current_paid_evaluation'] = {'kind': evaluated['evaluation_scope'], **deepcopy(evaluated)}
            count = evaluated['evaluated_rows']
            facts['current_paid_evaluation'].update(
                evaluated_response_row_range=[0, count - 1] if count else [],
                unscored_response_row_range=[count, len(self.rows) - 1] if count < len(self.rows) else [])
        for index, result in enumerate(self.operator_results):
            if result.get('operator') != 'refit':
                continue
            split = result.get('split', {})
            validation = result.get('validation', {})
            facts[f'refit:{index}'] = {
                'kind': 'paid_trajectory_validation', 'operator_index': index,
                'source_model': result.get('source_model', result.get('arguments', {}).get('model')),
                'tested_model': result.get('tested_model'), 'status': result.get('status'),
                'validation_mse': validation.get('mse'),
                'train_rows': split.get('train_rows'), 'holdout_rows': split.get('holdout_rows'),
                'validation_records': deepcopy(split.get('validation_records', [])),
                'source': {'file': 'research_state.json', 'json_path': ['operator_results', index]}}
        for identifier, entry in self.relation_research.items():
            facts[f'relation:{identifier}'] = {'kind': 'executed_structural_research',
                'question_id': identifier, **structural_summary(entry)}
        analyses = self._executed_python_analyses()
        cited = cited_ranges('analysis')
        selected = {index for index, _ in analyses[-2:]}
        # Preserve the author's currently cited records, not the author's claims
        # about them. Intersect ranges with real analyses rather than expanding
        # a range supplied in prose into synthetic evidence identifiers.
        selected.update(index for index, _ in analyses
                        if any(start <= index <= end for start, end in cited))
        # A changing notebook is an interpretation, not the only index of paid
        # work. Keep the active inquiry's actual executions and those of its
        # ancestors without selecting a scientific conclusion or other children.
        by_id = {question['id']: question for question in self.questions}
        lineage = set()
        identifier = self._active_question_id
        while identifier in by_id and identifier not in lineage:
            lineage.add(identifier)
            identifier = by_id[identifier].get('parent_id')
        selected.update(index for index, event in analyses
                        if event['feedback'].get('research_question_id') in lineage)
        returned_analyses = set()
        analysis_indices = {index for index, _ in analyses}
        for question in self.questions:
            returned = (question.get('research_return') or question.get('review_return')
                        or question.get('representation_return') or question.get('structural_return'))
            if (not returned or (question.get('origin') not in {
                    'fresh_confirmation_failure', 'representation_research', 'structural_proposal'}
                    and not question.get('research_return'))):
                continue
            parent_id = returned.get('parent_id', question.get('assignment', question).get('parent_id'))
            if parent_id not in lineage:
                continue
            evaluation = returned.get('evaluation')
            if evaluation is None and question.get('origin') == 'structural_proposal':
                evaluation = extract_existing_return_evaluation(self, returned['event_indices'])
            indices = [index for index in returned['event_indices'] if index in analysis_indices]
            returned_analyses.update(indices)
            facts[f"research_return:{question['id']}"] = {
                'kind': 'executed_research_return', 'question_id': question['id'],
                'assignment': deepcopy(returned.get('assignment') or question['assignment']),
                'parent_id': parent_id,
                'status': returned['status'], 'artifacts': deepcopy(returned['artifacts']),
                'analysis_refs': [f'analysis:{index}' for index in indices],
                'revision_refs': [f'revision:{index}' for index in returned.get('revision_indices', [])],
                **({'evaluation': deepcopy(evaluation)} if evaluation is not None else {}),
                'source': {'file': 'research_state.json',
                           'json_path': ['questions', self.questions.index(question)]}}
            for index in returned.get('revision_indices', []):
                record = self.revisions[index]
                facts[f'revision:{index}'] = revision_fact(record)
        selected.update(returned_analyses)
        for index, event in analyses:
            if index not in selected:
                continue
            feedback = event['feedback']
            fact = {'kind': 'executed_python_analysis', 'event_index': index,
                **{key: deepcopy(feedback[key]) for key in
                   ('returncode', 'timeout', 'error', 'status', 'executed_path',
                    'execution_seconds', 'remaining_execution_seconds',
                    'research_question_id') if key in feedback},
                'source': {'file': 'research_context.json', 'json_path': ['events', index, 'feedback']}}
            for key in ('stdout', 'stderr'):
                if key in feedback:
                    text = feedback[key]
                    for suffix in ('file', 'characters', 'truncated'):
                        name = f'{key}_{suffix}'
                        if name in feedback:
                            fact[name] = deepcopy(feedback[name])
                    if len(text) <= 4000:
                        fact[key] = text
                    elif feedback.get(f'{key}_file'):
                        fact[key] = {
                            'full_content_file': feedback[f'{key}_file'],
                            'characters': feedback.get(f'{key}_characters', len(text))}
                    else:
                        fact[key] = {
                            'characters': len(text), 'file': 'research_context.json',
                            'json_path': ['events', index, 'feedback', key]}
                    if index in returned_analyses and len(text) > 4000:
                        fact[key].update(preview=text[:2000], tail=text[-1200:])
            facts[f'analysis:{index}'] = fact
        return {'current_paid_rows': len(self.rows), 'facts': facts,
                'checked_relations': checked_relations,
                'scope': 'Executed records only. These facts do not establish the authored scientific explanation.'}

    def _attach_research_advice(self, action, arguments, result, laboratory):
        # This ordinary-Python condition exposes no shared instrument solver.
        # Preserve executed feedback without proposing the old meter workflow.
        result.pop("available_research_action", None)
        self._current_research_advice = None
        return result

    def _registered_research_programs(self, manifest):
        # Submission bindings name the executed method but omit its registered
        # calling convention and paid support. Use the same merge online and
        # during export so a failed submission cannot erase that information.
        registered = {entry["path"]: entry for entry in
                      self.knowledge.get("manifest", {}).get("research_programs", [])}
        return [{**deepcopy(registered.get(entry.get("path"), {})), **entry}
                for entry in manifest.get("research_programs", [])]

    def _method_continuation(self, entry):
        if self.model is not None or not entry["current_executions"]:
            return None
        latest = entry["current_executions"][-1]
        feedback = latest["feedback"]
        if feedback.get("status") == "method_review_required":
            review = feedback['review']
            if review.get('kind') == 'research_evidence_extended':
                review = native_evidence_extension_review(self, entry['path'])
                if review is None:
                    return None
            return {"status": "method_review_required", "method_step": latest["step"],
                "review": deepcopy(review), "revision_path": entry["path"],
                "current_paid_rows": len(self.rows)}
        if feedback.get("status") == "method_stopped":
            review = native_evidence_extension_review(self, entry['path'])
            if review is not None:
                return {'status': 'method_review_required', 'method_step': latest['step'],
                    'review': review, 'revision_path': entry['path'],
                    'current_paid_rows': len(self.rows)}
        confirmation = feedback.get("fresh_confirmation") or {}
        if confirmation.get("passed") is not False:
            return None
        proposal = latest.get("proposal") or {}
        scored = confirmation.get("returncode") == 0 and confirmation.get("mse") is not None
        return {
            "status": "prediction_refuted" if scored else "confirmation_execution_failed",
            "submitted_model": feedback.get("method_submission") or proposal.get("model")
                or proposal.get("arguments", {}).get("model"),
            "submitted_candidate": deepcopy(feedback.get("submitted_candidate")),
            "method_step": latest["step"],
            "confirmation": {key: deepcopy(confirmation[key]) for key in
                ("passed", "mse", "rows", "mse_threshold", "returncode", "timeout", "stderr", "residuals")
                if key in confirmation},
            "new_paid_observation_indices": deepcopy(confirmation.get("observation_indices", [])),
            "current_paid_rows": len(self.rows),
            "resume_action": deepcopy(entry["execute_action"]),
            "revision_path": entry["path"],
        }

    def _program_state(self):
        state = super()._program_state()
        state["experiment_comparisons"] = self.experiment_comparisons
        state["online_research"] = self._online_research()
        state["observable_models"] = self.observable_models
        state["conditional_models"] = self.conditional_models
        state["model_dependencies"] = self.model_dependencies
        state["research_candidates"] = self.research_candidates
        state["research_input_bindings"] = self.research_input_bindings
        state['relation_research'] = self.relation_research
        # The existing state writer consumes this value synchronously, as it
        # already does for questions, costs and operator_results in base layers.
        state['theory_index'] = self.theory_index
        state['research_mission'] = self.research_mission
        state['research_agenda'] = self.research_agenda
        if self.theory_consolidation:
            state['theory_consolidation'] = {'file': LIBRARY_PATH}
        return state

    def _prepare_program_state(self):
        super()._prepare_program_state()
        manifest = self.manifest or self.knowledge.get("manifest", {})
        path = self.workspace / "shared/knowledge.json"
        if path.is_file():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(current, dict):
                    manifest = current
            except json.JSONDecodeError:
                pass  # An unfinished manifest does not erase the inherited entry.
        methods = []
        for method in self._registered_research_programs(manifest):
            if not (self.workspace / method["path"]).is_file():
                continue
            entry = deepcopy(method)
            entry["execute_action"] = ({"action": "run", "arguments": {
                "operator": "procedure", "path": method["path"],
                "max_steps": method["max_steps"], "confirm": True}}
                if "max_steps" in method else
                {"action": "query", "arguments": {"path": method["path"]}})
            executions = [(index, outcome) for index, outcome in enumerate(self.method_results)
                          if outcome["path"] == method["path"]]
            entry["current_executions"] = [outcome for _, outcome in executions]
            entry["submitted_candidates"] = [
                {"method_step": outcome["step"], **deepcopy(outcome["feedback"]["submitted_candidate"]),
                 "confirmation": {key: deepcopy(outcome["feedback"]["fresh_confirmation"][key])
                    for key in ("passed", "mse", "returncode", "timeout", "observation_indices")}}
                for outcome in entry["current_executions"]
                if "submitted_candidate" in outcome["feedback"]]
            entry["source_unchanged"] = self._provenance(method["path"])["inherited_source_unchanged"]
            continuation = self._method_continuation(entry)
            if continuation is not None:
                entry["continuation"] = continuation
            # The state already owns the full chronology. Persist the same
            # compact view used by provider context, with references to that
            # canonical state rather than another full copy in method_decision.
            entry["current_executions"] = []
            for index, outcome in executions:
                view = {}
                for key, value in outcome.items():
                    location = ["method_results", index, key]
                    if key == 'feedback' and isinstance(value, dict):
                        view[key] = {field: (deepcopy(item) if len(json.dumps(item)) <= 12000 else
                            _saved_context_reference(item, [*location, field], 'research_state.json'))
                            for field, item in value.items()}
                    else:
                        view[key] = (deepcopy(value) if len(json.dumps(value)) <= 12000 else
                            _saved_context_reference(value, location, 'research_state.json'))
                entry["current_executions"].append(view)
            methods.append(entry)
        native = native_condition_method(self)
        self._save("method_decision.json", json.dumps({
            "research_options": [native] if native is not None else [],
            "methods": methods}, indent=2))

    def _base_result_files(self):
        # This class replaces the base outgoing knowledge and changed-file view
        # below. Construct its actual portable/workspace export once, there.
        return {}

    def result(self, status):
        return self._result_with_snapshot(status, deepcopy)

    def checkpoint_result(self, status):
        """Build an immediately written checkpoint, borrowing live research data.

        Unlike result(), this transient value must be serialized before the
        agent continues. Retained results still receive an independent snapshot.
        """
        return self._result_with_snapshot(status, lambda value: value)

    def _result_with_snapshot(self, status, snapshot):
        if self.theory_index or getattr(self, '_theory_index_source_ref', None):
            save_theory_index(self)
        result = super()._result_with_snapshot(status, snapshot)
        result["experiment_comparisons"] = snapshot(self.experiment_comparisons)
        result["observable_models"] = snapshot(self.observable_models)
        result["conditional_models"] = snapshot(self.conditional_models)
        result["model_dependencies"] = snapshot(self.model_dependencies)
        result["research_candidates"] = snapshot(self.research_candidates)
        result["research_input_bindings"] = snapshot(self.research_input_bindings)
        result['relation_research'] = snapshot(self.relation_research)
        result['theory_index'] = snapshot(self.theory_index)
        result['research_mission'] = snapshot(self.research_mission)
        result['research_agenda'] = snapshot(self.research_agenda)
        if self.theory_consolidation:
            result['theory_consolidation'] = {'file': LIBRARY_PATH}
        result['llm_inquiries'] = snapshot(self.llm_inquiries)
        result['inquiry_stack'] = snapshot(self._inquiry_stack)
        # A failed or unfinished source may already have authored a useful
        # manifest. The old result only retains self.manifest after submission.
        manifest = deepcopy(self.manifest or self.knowledge.get("manifest", {}))
        manifest_paths = ["shared/knowledge.json"]
        model_paths = [self.model] if self.model else []
        evaluated = self._evaluated_summary()
        if evaluated:
            model_paths.append(evaluated['model'])
            model_paths.append(evaluated['source_model'])
        for event in self.events:
            arguments = event.get("arguments", {})
            if arguments.get("manifest"):
                manifest_paths.append(arguments["manifest"])
        for name in manifest_paths:
            path = self.workspace / name
            if path.is_file():
                try:
                    authored = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    # Keep the unfinished file itself below; checkpointing must
                    # not discard a source merely because it was mid-revision.
                    continue
                if isinstance(authored, dict):
                    manifest = authored

        # A procedure's submission creates execution bindings, while the
        # inherited manifest records how that same published method is called.
        # Keep the newly submitted hypothesis and backfill its method metadata
        # rather than replacing the publication with a path-only binding.
        for field in ("hypotheses", "research_programs"):
            additions = self.manifest.get(field, [])
            if not additions:
                continue
            entries = manifest.setdefault(field, [])
            paths = {entry.get("path") for entry in entries}
            entries.extend(deepcopy(entry) for entry in additions
                           if entry.get("path") not in paths)
        if manifest.get("research_programs"):
            manifest["research_programs"] = self._registered_research_programs(manifest)

        files = self._outgoing_files_for_result(status, manifest, manifest_paths, model_paths)
        lessons = [*snapshot(self.knowledge.get("lessons", []))]
        if self.lesson:
            lessons.append(self.lesson)
        result["outgoing_knowledge"] = {
            "files": files,
            "manifest": deepcopy(manifest),
            "question_traces": [*snapshot(self.knowledge.get("question_traces", [])),
                                *snapshot(self.questions)],
            "lessons": lessons,
        }
        if self.theory_index or getattr(self, '_theory_index_source_ref', None):
            result['outgoing_knowledge']['files'].pop(INDEX_PATH, None)
            result['outgoing_knowledge']['theory_index'] = {
                'file': str(self.workspace / INDEX_PATH), 'json_path': []}
        if self.theory_consolidation:
            result['outgoing_knowledge']['files'].pop(LIBRARY_PATH, None)
            result['outgoing_knowledge']['theory_consolidation'] = {
                'file': str(self.workspace / LIBRARY_PATH), 'json_path': []}
        for key in ("research_roles", "source_catalog", "archive_instructions"):
            if key in self.knowledge:
                result["outgoing_knowledge"][key] = snapshot(self.knowledge[key])
        result["inherited_files_changed"] = sorted(
            name for name in self.inherited_files
            if name in files and files[name] != self.knowledge["files"][name])
        result["knowledge_protocol"] = "common_executable_archive"
        return result

    def _outgoing_files_for_result(self, status, manifest, manifest_paths, model_paths):
        """Materialize ordinary portable exports when the consumer needs files."""
        if status == "researching":
            # Same-workspace checkpoints retain snapshots on disk. Publishing
            # below still transfers their fitting views to independent tasks.
            names = set(self.files) | set(manifest_paths) | set(model_paths) | {
                "observations.json", "model_observations.json", "laboratory_observations.json"}

            def authored_paths(value):
                if isinstance(value, dict):
                    if isinstance(value.get("path"), str):
                        names.add(value["path"])
                    for item in value.values():
                        authored_paths(item)
                elif isinstance(value, list):
                    for item in value:
                        authored_paths(item)

            authored_paths(manifest)
            referenced_files = ({INDEX_PATH} if self.theory_index
                or getattr(self, '_theory_index_source_ref', None) else set())
            if self.theory_consolidation:
                referenced_files.add(LIBRARY_PATH)
            # These files are published by reference below; their complete
            # text would otherwise be read and discarded on every checkpoint.
            files = self._shared_files(excluded=referenced_files)
            files.update({name: (self.workspace / name).read_text(encoding="utf-8")
                for name in names if name not in referenced_files
                and "_evaluated_model" not in Path(name).parts
                and (self.workspace / name).is_file()
                and (self.workspace / name).name not in _HOST_RECORD_FILES})
        else:
            files = self._result_files(manifest, manifest_paths, model_paths)
        return files

    def _result_files(self, manifest, manifest_paths, model_paths):
        """Export executable source research for later independent consumers."""
        frozen_directories = {'_evaluated_model', '_target_work', '_experiment_models',
                              '_submitted_models', '_observable_models'}
        names = {name for name in self.files
                 if not frozen_directories.intersection(Path(name).parts)}
        names.update(manifest_paths)
        selected_paths = set(model_paths)
        declared_contexts = set()
        for value in self.knowledge.get('research_roles', {}).values():
            if isinstance(value, str):
                selected_paths.add(value)
            elif isinstance(value, list):
                selected_paths.update(value)
        for entry in self.knowledge.get('source_catalog', []):
            if entry.get('model'):
                selected_paths.add(entry['model'])
            if entry.get('directory'):
                declared_contexts.add(self.workspace / entry['directory'])

        def referenced_paths(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "path" and isinstance(item, str):
                        selected_paths.add(item)
                    referenced_paths(item)
            elif isinstance(value, list):
                for item in value:
                    referenced_paths(item)

        referenced_paths(manifest)

        collected_contexts = set()

        def collect(context):
            if context in collected_contexts:
                return
            collected_contexts.add(context)
            # Prune before descending: old comparisons can contain repeated
            # copies of the entire workspace. Selected contexts are added below.
            for directory, subdirs, filenames in os.walk(context):
                subdirs[:] = [name for name in subdirs
                    if name not in frozen_directories
                    and not _model_context_ignored(Path(directory) / name, context)]
                for filename in filenames:
                    path = Path(directory) / filename
                    if (path.suffix in {'.py', '.json', '.md', '.txt'}
                            and not _model_context_ignored(path, context)):
                        names.add(str(path.relative_to(self.workspace)))

        catalog_contexts = set()

        def include_catalog(context):
            if context in catalog_contexts:
                return
            catalog_contexts.add(context)
            path = context / 'source_catalog.json'
            if path.is_file():
                try:
                    catalog = json.loads(path.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    return
                selected_paths.update(str((context / entry['model']).relative_to(self.workspace))
                                      for entry in catalog if entry.get('model'))
                declared_contexts.update(context / entry['directory']
                                         for entry in catalog if entry.get('directory'))

        collect(self.workspace)
        include_catalog(self.workspace)
        visited = set()
        while selected_paths or declared_contexts:
            if declared_contexts:
                context = declared_contexts.pop()
                collect(context)
                include_catalog(context)
                continue
            name = selected_paths.pop()
            if name in visited:
                continue
            visited.add(name)
            names.add(name)
            context, model = python_model_context(self.workspace, name)
            if context != self.workspace:
                collect(context)
            include_catalog(context)
            # Reuse the executable model's declared dependency contract. An
            # explicitly bound frozen child keeps its own fitting context.
            bindings_file = context / 'model_bindings.json'
            if bindings_file.is_file():
                bindings = json.loads(bindings_file.read_text(encoding='utf-8')).get(model, {})
                selected_paths.update(str((context / child).relative_to(self.workspace))
                                      for child in bindings.values())
        referenced_files = ({INDEX_PATH} if self.theory_index
            or getattr(self, '_theory_index_source_ref', None) else set())
        if self.theory_consolidation:
            referenced_files.add(LIBRARY_PATH)
        return {name: (self.workspace / name).read_text(encoding="utf-8")
                for name in sorted(names) if name not in referenced_files
                and (self.workspace / name).is_file()
                and not _model_context_ignored(self.workspace / name,
                    python_model_context(self.workspace, name)[0])}


class FlatScientificRevisionAgent(CommonArchiveInquiryAgent):
    """Keep the designated failure review in the parent's notes and event scope."""

    def _flat_review(self):
        return next((q for q in reversed(self.questions)
                     if q.get('flat_review', {}).get('status') == 'running'), None)

    def _dispatch_scientific_revision(self, question, parent_event_index):
        question['flat_review'] = {
            'status': 'running', 'start_calls': self.costs['model_calls'],
            'call_limit': min(self.costs['model_calls'] + question.get('review_model_calls', 8),
                              self.config.max_model_calls - 1),
            'parent_event_index': parent_event_index, 'parent_question': question.get('parent_id'),
            'event_start': len(self.events), 'stop_requested': False}
        self._active_question_id = question['id']
        self.events[parent_event_index]['feedback']['scientific_revision'] = {
            'status': 'inquiry_started', 'organization': 'flat_review',
            'question': deepcopy(question), 'assignment': deepcopy(question['assignment'])}

    def _settle_flat_review(self):
        question = self._flat_review()
        if question is None or self._inquiry_stack:
            return
        frame = question['flat_review']
        same_assignment = all(question.get(key) == question['assignment'].get(key)
                              for key in ('question', 'parent_id', 'depends_on'))
        states = {q['id']: q.get('status') for q in self.questions}
        pending = any(states.get(identifier) != 'supported'
                      for identifier in question.get('depends_on', []))
        if frame['stop_requested']:
            status = 'inquiry_finished'
        elif same_assignment and question.get('status') in {'supported', 'refuted'} and not pending:
            status = 'answered'
        elif self.costs['model_calls'] >= frame['call_limit']:
            status = 'model_call_budget_exhausted'
        else:
            return
        indices = list(range(frame['event_start'], len(self.events)))
        artifacts = []
        for index in indices:
            event = self.events[index]
            args, feedback = event.get('arguments', {}), event.get('feedback', {})
            declared = args.get('artifacts', [])
            for path in [feedback.get('written'), args.get('model'),
                         *(declared if isinstance(declared, list) else [])]:
                if (isinstance(path, str) and (self.workspace / path).is_file()
                        and path not in artifacts):
                    artifacts.append(path)
        notes = self.workspace / 'research_notes.md'
        result = {'operator': 'inquire', 'organization': 'flat_review', 'status': status,
            'question': {key: deepcopy(value) for key, value in question.items() if key != 'flat_review'},
            'assignment': deepcopy(question['assignment']),
            'model_calls': self.costs['model_calls'] - frame['start_calls'],
            'notes': notes.read_text() if notes.exists() else '',
            'artifacts': artifacts, 'event_indices': indices}
        frame['status'] = status
        self._record_inquiry_return(frame, result)
        self._active_question_id = frame['parent_question']

    def _before_model_turn(self, laboratory):
        self._settle_flat_review()
        return super()._before_model_turn(laboratory)

    def _after_model_turn(self, laboratory):
        if self._flat_review() is None:
            return super()._after_model_turn(laboratory)
        # Match the child review: execution errors are returned to its model,
        # without adding automatic debug sessions to the flat condition only.
        self._settle_inquiries()
        self._settle_flat_review()

    def _start_inquiry(self, arguments, *, parent_event_index=None):
        result = super()._start_inquiry(arguments, parent_event_index=parent_event_index)
        question = self._flat_review()
        if question is not None:
            frame = self._inquiry_stack[-1]
            frame['call_limit'] = min(frame['call_limit'], question['flat_review']['call_limit'] - 1)
        return result

    def _execute(self, action, arguments, laboratory):
        question = self._flat_review()
        if question is not None and not self._inquiry_stack:
            if action == 'finish':
                if 'notes' in arguments:
                    self._save('research_notes.md', arguments['notes'])
                result = self._finish_local_research(arguments, laboratory)
                if result['status'] == 'inquiry_finished':
                    question['flat_review']['stop_requested'] = True
                return result
            if not arguments.get('research_question') and not arguments.get('question'):
                arguments = {**arguments, 'research_question': {'id': question['id']}}
        return super()._execute(action, arguments, laboratory)

    def _request(self, laboratory):
        request = super()._request(laboratory)
        question = self._flat_review()
        if question is not None and not self._inquiry_stack:
            frame = question['flat_review']
            request['scientific_revision_review'] = {
                'question_id': question['id'], 'organization': 'flat_review',
                'remaining_model_calls': frame['call_limit'] - self.costs['model_calls']}
            request['task_interface'] = (
                'Resolve scientific_revision_assignment using the current paid evidence and tools, '
                'then return your executable work and its evidence to the caller. The apparatus '
                'task below is the enclosing prediction task. Local finish only returns to the '
                'parent; it neither requests response confirmation nor proves the research hypothesis. '
                'An explicitly returned model in local finish is evaluated on paid evidence first; '
                'include component and view for an authored local relation. Execution failure '
                'returns to this same researcher for repair.\n\n'
                + request['task_interface'])
        return request
