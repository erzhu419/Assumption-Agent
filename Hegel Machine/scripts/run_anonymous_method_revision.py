#!/usr/bin/env python3
"""Revise a published method from one paid development counterexample.

Source artifacts stay read only. All target slots are fixed before authoring;
both frozen methods finish research before either receives terminal evaluation.
"""

import argparse
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.anonymous_research_archive import load_research_archive
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.code_agent import (
    _copy_model_file, archived_paid_trace_references,
    CodexCodeProvider, _copy_model_context, python_model_context_matches)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, register_published_method
from hegel_machine.cognition.temporal_paired_practice import practice_execution_summary
from scripts.run_anonymous_method_learning import laboratory, trial_preparer, frozen_transfers, score_frozen
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_python_method_learning_study import LIMIT_COSTS, usage_summary
from scripts.run_temporal_counterexample_learning import supply_counterexample


COUNTEREXAMPLE_CASE = 'case_006'
COUNTEREXAMPLE = {'kind': 'hysteresis', 'seed': 62113038,
    'prefix_seed': 62213552, 'confirmation_seed': 62313561}


def protocol(source, *, model='gpt-6-luna', timeout_seconds=300., continue_author=None,
             repair_study=None, design_experiments=False, prospective_design=False,
             conditional_design=False):
    source = Path(source).resolve()
    if conditional_design:
        if prospective_design or design_experiments or repair_study or continue_author:
            raise ValueError('conditional design starts from one published source')
        old = json.loads((source / 'protocol.json').read_text())
        author = json.loads((source / 'author/training.json').read_text())
        if author['status'] != 'completed' or not author.get('method_source'):
            raise ValueError('conditional design requires a published method')
        return conditional_design_protocol(source, old, author, model=model,
                                           timeout_seconds=timeout_seconds)
    if prospective_design:
        if design_experiments or repair_study is not None or continue_author is not None:
            raise ValueError('prospective design revises one published method in a fresh study')
        old = json.loads((source / 'protocol.json').read_text())
        author = json.loads((source / 'author/training.json').read_text())
        if author['status'] != 'completed' or not author.get('method_source'):
            raise ValueError('prospective design requires the published acquisition method')
        return prospective_design_protocol(source, old, author, model=model,
                                           timeout_seconds=timeout_seconds)
    if design_experiments and (repair_study is not None or continue_author is not None):
        raise ValueError('experimental design starts one fresh author from the published method')
    if repair_study is not None and continue_author is not None:
        raise ValueError('the repair-study author has one fixed allowance; do not extend it')
    if continue_author is not None:
        path = Path(continue_author).resolve()
        previous = json.loads(path.read_text())
        if previous.get('method_source') is not None or previous['status'] not in {
                'provider_failed', 'runtime_interrupted', 'token_budget_exhausted', 'model_call_budget_exhausted'}:
            raise ValueError('continue an actual unpublished interrupted or exhausted author')
        plan = json.loads((path.parent.parent / 'protocol.json').read_text())
        if plan.get('repair_study'):
            raise ValueError('the repair-study author has one fixed allowance; do not extend it')
        if Path(plan['source']).resolve() != source:
            raise ValueError('continuation must retain the same published source archive')
        if previous['status'] == 'runtime_interrupted':
            if not plan.get('prospective_design'):
                raise ValueError('runtime interruption resumes the existing prospective design account')
            earlier = {'prior_author_phases': plan['prior_author_phases']}
            plan['runtime_snapshot'] = sorted(set(plan.get('runtime_snapshot', [])) | {
                'src/hegel_machine/cognition/interrupted_method_author.py'})
        else:
            earlier = json.loads((path.parent.parent / 'summary.json').read_text())
        plan.update(continued_author=str(path), model=model,
            prior_author_phases=[*earlier['prior_author_phases'], {
                'result': str(path), 'status': previous['status'],
                'costs': previous['costs'], 'usage': usage_summary(previous)}],
            continuation='Additional author allowance with the same seven practices and fixed targets. '
                'Restore every paid draft and failure. Reuse the original paid counterexample without '
                'reacquisition; connect the historical counterexample ledger and revision task. '
                'No fitted source, scientific diagnosis or repair is supplied by the host.')
        if plan.get('prospective_design'):
            if previous['status'] not in {'provider_failed', 'runtime_interrupted'}:
                raise ValueError('prospective design resumes interruptions within its fixed allowance')
            # Reuse the historical author-resumption accounting. Earlier method
            # learning is background cost, not spending from this design account.
            remaining = deepcopy(plan['author_limits'])
            plan.setdefault('author_total_limits', deepcopy(remaining))
            for limit, cost in LIMIT_COSTS:
                remaining[limit] = max(0, remaining[limit] - previous['costs'][cost])
            usage = usage_summary(previous)
            remaining['token_budget'] = max(0, remaining['token_budget'] - usage['known_tokens'])
            plan.update(author_limits=remaining,
                continuation='Resume the actual interruption with the same scientific '
                    'protocol and remaining author allowance. Restore paid practices and the exact '
                    'unfinished program without replay. No host scientific correction is supplied.',
                continuation_usage={'known_tokens_charged': usage['known_tokens'],
                    'missing_usage_calls': usage['missing_usage_calls'],
                    'scope': 'Unknown token usage stays unknown; no estimate is substituted.'})
        plan['provider'] = {**plan['provider'], 'timeout_seconds': timeout_seconds}
        return plan
    old = json.loads((source / 'protocol.json').read_text())
    author = json.loads((source / 'author' / 'training.json').read_text())
    if author['status'] != 'completed' or not author.get('method_source'):
        raise ValueError('revision requires an actually published source method')
    if design_experiments:
        return experiment_design_protocol(source, old, author, model=model)
    if repair_study is not None:
        return repair_protocol(source, old, repair_study, model=model)
    practice = deepcopy(old['practice_cases'])
    if COUNTEREXAMPLE_CASE in practice:
        raise ValueError('this development counterexample is already registered')
    practice[COUNTEREXAMPLE_CASE] = deepcopy(COUNTEREXAMPLE)
    targets = deepcopy(old['transfer_cases'])
    first_by_family = {}
    for case in old['transfer_cases']:
        first_by_family.setdefault(case['kind'], case)
    for index, original in enumerate(first_by_family.values()):
        seed = original['seed']
        for prefix in range(8):
            targets.append({'id': f'new_object_{index:03d}_prefix_{prefix:03d}',
                'kind': original['kind'], 'seed': seed + 3100033,
                'prefix_seed': seed + 4100039 + prefix,
                'confirmation_seed': seed + 5100077 + prefix,
                'evaluation_seed': seed + 6100091 + prefix})
    return {'study': 'anonymous_published_method_counterexample_revision',
        'source': str(source), 'model': model,
        'provider': {'backend': 'codex_cli', 'timeout_seconds': timeout_seconds,
                     'reasoning_effort': 'medium'},
        'author_limits': {**old['author_limits'], 'max_model_calls': 24,
                          'token_budget': 1500000, 'observation_budget': 6000},
        'limits': deepcopy(old['limits']), 'prefix_rows': old['prefix_rows'],
        'practice_cases': practice, 'transfer_cases': targets,
        'old_target_ids': [case['id'] for case in old['transfer_cases']],
        'evaluation': deepcopy(old['evaluation']),
        'counterexample': {'case': COUNTEREXAMPLE_CASE, **deepcopy(COUNTEREXAMPLE),
            'origin': 'Development failure found at the sixth ordinary random prefix on the '
                'previously practiced object. Selected after observing that failure; not held out.',
            'confirmation_rule': 'The first reproduction uses this exact confirmation seed. '
                'Later ordinary practices retain the existing seed plus practice-count rule.'},
        'restoration': 'Restore all earlier paid source drafts and practice results without replay. '
            'The published program is a revisable starting point, not publication on the new case.',
        'comparison': 'Same original executable archive and limits for both methods. Retain the old '
            'nine targets and add three new objects with eight preregistered ordinary prefixes each. '
            'No live consumer or externally selected target repairs; each frozen program acts itself.',
        'terminal_order': 'Finish both methods on all target slots before generating terminal labels.',
        'failure_policy': 'Retain paid failed authoring and reproduction. Without publication, all '
            'paired target slots remain unavailable; do not substitute an unpublished draft.',
        'scope': 'One development counterexample and one author continuation. Prefix repetitions '
            'on each object are not independent discoveries; the retained old targets are revisits.'}


def experiment_design_protocol(source, old, author, *, model):
    """Reuse a successful method while removing confirmation as its research teacher."""
    old_ids = old['old_target_ids']
    retained = [deepcopy(case) for case in old['transfer_cases'] if case['id'] in old_ids]
    targets = list(retained)
    for block, seed in enumerate((91231031, 91232040)):
        for kind in ('recurrent', 'finite2'):
            targets.append({'id': f'probe_target_{len(targets) - len(retained):03d}',
                'parameter_pair': block, 'kind': kind, 'seed': seed, 'apparatus_seed': seed,
                'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
                'evaluation_seed': seed + 300007})
    historical = json.loads((source / 'summary.json').read_text())
    return {'study': 'recurrent_experiment_design_method_transfer', 'source': str(source),
        'design_experiments': True, 'model': model,
        'provider': {'backend': 'codex_cli', 'timeout_seconds': 600., 'reasoning_effort': 'medium'},
        'author_limits': {**old['author_limits'], 'max_model_calls': 96,
            'token_budget': 20_000_000, 'observation_budget': 36_000,
            'execution_seconds': 7200., 'compute_budget': 4_000_000},
        'limits': {**old['limits'], 'observation_budget': 1152, 'execution_seconds': 3600.},
        'prefix_rows': old['prefix_rows'],
        'practice_cases': {f'probe_case_{i:03d}': deepcopy(case)
            for i, case in enumerate(old['practice_cases'].values())},
        'previous_practice_cases': {f'probe_case_{i:03d}': name
            for i, name in enumerate(old['practice_cases'])},
        'transfer_cases': targets, 'old_target_ids': [case['id'] for case in retained],
        'evaluation': deepcopy(old['evaluation']),
        'prior_author_phases': [*historical.get('prior_author_phases', []), {
            'result': str(source / 'author/training.json'), 'status': author['status'],
            'costs': author['costs'], 'usage': usage_summary(author)}],
        'restoration': 'Restore the published source and every earlier paid practice as experience. '
            'Renamed practice cases require new support under the single-submission protocol; '
            'old confirmation-assisted successes cannot establish that support.',
        'author_experience': 'Use the inherited method and its saved practice traces, including '
            'failures. Do not append or repeat the eight earlier repair investigations. Only the '
            'unchanged original archive and published self-contained method enter any target.',
        'practice_protocol': 'All six practices allow the method to author experiments and acquire '
            'evidence. The first executable model submission ends program research and receives '
            'one ordinary 96-row fresh confirmation. Its labels never reenter the same program. '
            'The author may inspect completed practices to revise the next method draft.',
        'comparison': 'Both methods receive identical K0, prefixes, tools and limits. Six original '
            'targets use their original confirmation-assisted protocol to check retained behavior. '
            'Four new targets on two parameter blocks use the same single-submission protocol '
            'as practice, with no LLM calls or confirmation feedback during method execution.',
        'outcome_protocol': 'Record actual competing predictions, authored proposed experiments, '
            'paid acquisition, hypothesis consequences and subsequent choices. Publication requires '
            'predictive success and existing reuse/revision requirements, never mandatory tool counts. '
            'Prediction success alone does not establish autonomous experimental discrimination.',
        'terminal_order': 'Finish both frozen research cohorts before generating any terminal labels.',
        'failure_policy': 'One fixed author allowance. Without publication retain every failure and '
            'leave paired target slots unavailable; do not transfer an unpublished draft.',
        'scope': 'One prompted method revision within two known apparatus families, six old-target '
            'revisits and two new parameter blocks. The new experiment tests portable self-authored '
            'experimental design, not open-world discovery or an isolated recursive advantage.',
        'runtime_snapshot': sorted({str(path.relative_to(ROOT))
            for path in (ROOT / 'src/hegel_machine').rglob('*.py')} | set(old.get('runtime_snapshot', [])) | {
            'scripts/run_anonymous_method_revision.py', 'scripts/run_anonymous_method_learning.py',
            'scripts/run_anonymous_online_research.py'})}


def attach_experiment_design_task(author):
    task = ('Revise the inherited research method so that it designs informative experiments.\n'
        'Use the inherited source, its actual successful and failed practices, and source_catalog.json.\n'
        'Construct runnable competing explanations when warranted by paid evidence, compare their\n'
        'predictions before acquisition, and author experiments and readouts that can distinguish them.\n'
        'Use actual observations to retain or revise explanations and subsequent research choices.\n'
        'The shared compare_experiments and revision tools are available; their use is your decision.\n'
        'Keep useful old behavior and make your experiment-design method portable across practice cases.\n'
        'The inherited programs contain prior authored scientific knowledge; no additional hypotheses,\n'
        'fitting formulas, candidate experiment menu or scientific readout are supplied for this revision.\n'
        'Every listed probe_case requires fresh paid support. Old trials remain research experience\n'
        'but were run under a different confirmation protocol and do not count as current support.\n'
        'During these practices the FIRST executable model submission ENDS program research. One\n'
        '96-row blind confirmation follows; its data cannot drive another invocation of that program.\n'
        'A nested procedure returns a candidate proposal to its parent without final confirmation;\n'
        'the parent may continue research before making its own final submission.\n'
        'Author your needed experiments before submission. You may inspect a completed practice and\n'
        'revise or retain the next draft using its outcome. A failed submission is a failed practice.\n'
        'Only the final self-contained method and original archive transfer, not practice-fitted files.\n\n')
    author.common_interface = task + 'AUTHOR ACTIONS:' + author.common_interface.split('AUTHOR ACTIONS:', 1)[1]
    author.common_interface = author.common_interface.replace(
        'The initial method is only an unresolved placeholder. You author its decisions,\n'
        'predictors, parameter estimation and experiments using ordinary Python.',
        'The inherited method and saved drafts are revisable Python programs with paid histories.')
    author.common_interface = author.common_interface.replace(
        'Your program\ncan use actual feedback to repair a failure before that limit.',
        'Your program\ncan use acquisition and analysis feedback until its first model submission.')
    author.common_interface = author.common_interface.replace(
        'Failed confirmation becomes paid evidence on the next invocation.',
        'Confirmation ends this practice; the program is not invoked again with those labels.')
    author.common_interface = author.common_interface.replace(
        'Failed confirmation contributes paid counterexamples, visible on the next invocation.',
        'Confirmation ends this practice; its counterexamples are available to the method author only.')


def prospective_design_protocol(source, old, author, *, model, timeout_seconds):
    """Separate the design objective and choice intervention from predictor success."""
    plan = experiment_design_protocol(source, old, author, model=model)
    retained = [deepcopy(case) for case in old['transfer_cases'] if case['id'] in old['old_target_ids']]
    targets = list(retained)
    for block, seed in enumerate((101231031, 101232040)):
        for kind in ('recurrent', 'finite2'):
            targets.append({'id': f'design_target_{len(targets)-len(retained):03d}',
                'parameter_pair': block, 'kind': kind, 'seed': seed, 'apparatus_seed': seed,
                'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
                'evaluation_seed': seed + 300007})
    plan.update(study='prospective_experiment_design_method_transfer', prospective_design=True,
        method_steps=32, transfer_cases=targets,
        practice_cases={f'design_case_{i:03d}': deepcopy(case)
                        for i, case in enumerate(old['practice_cases'].values())},
        previous_practice_cases={f'design_case_{i:03d}': name
                                for i, name in enumerate(old['practice_cases'])},
        comparison='Three frozen arms: the unchanged published fixed-acquisition method; '
            'the newly authored method; and that exact new method with uniform experiment choice '
            'from its own equal-cost offered pool. Both new-method arms keep the same candidate '
            'generation, fitting, tools and evidence interface. Actual subsequent evidence and '
            'decisions may differ. All ten targets use the same single-submission protocol.',
        practice_protocol='Six source cases reacquire support, with 72 prefix rows, up to 192 '
            'author-chosen experimental rows and one final 96-row confirmation. Up to 32 program '
            'invocations allow sequential forecasts and acquisition. Final labels never reenter '
            'the program. Old successes remain available and do not qualify as new practice.',
        outcome_protocol='Retain predictive publication separately from the experimental-design '
            'objective. The latter requires a current-source supported-rival forecast, actual '
            'discriminating acquisition and authored evidence readout before submission, followed '
            'by accepted prediction. Refuting all candidate instances and refitting is allowed. '
            'No tool-call quota or accuracy improvement over the fixed method is required.',
        failure_policy='One author allowance. If the design objective remains unresolved, retain '
            'the prediction-supported source and every failure but do not label it a design '
            'method or replace the existing successful baseline. Target comparisons stay unavailable.',
        scope='One prompted design-method revision, six seen-source practices, six retention '
            'revisits and two new parameter blocks in known families. The randomized intervention '
            'tests experiment choice within the same new program. Equal final scores are valid; '
            'no claim of general autonomous discovery or recursive organization advantage.')
    plan['provider']['timeout_seconds'] = timeout_seconds
    plan['limits'].update(observation_budget=360, execution_seconds=7200.)
    plan['author_limits']['execution_seconds'] = 18000.
    plan['runtime_snapshot'] = sorted(set(plan['runtime_snapshot']) | {
        'scripts/experimental_design_author.py', 'scripts/experimental_design_control.py'})
    return plan


def attach_prospective_design_task(author):
    attach_experiment_design_task(author)
    author.common_interface = (
        'CURRENT RESEARCH DELIVERABLE: a prospective experiment-design method.\n'
        'The inherited method already predicts successfully after a fixed acquisition suite.\n'
        'Preserve that useful work; final predictive accuracy alone does not answer this task.\n'
        'Form executable alternatives from current paid evidence BEFORE selecting acquisition.\n'
        'Use prospective differences to choose among experiments you construct, then use their\n'
        'actual outcomes to retain, refit or revise explanations and subsequent research choices.\n'
        'Alternatives may be different supported parameterizations within one representation.\n'
        'If all predicted outcomes fail, revise those instances; that alone does not exclude\n'
        'their entire representation family. An identifiable case may directly reuse knowledge.\n'
        'Use the existing forecast/revision/readout interface to expose the actual scientific\n'
        'sequence. No fitting formulas, new scientific alternatives or experiment menus are added.\n'
        'experiment_design_feedback.json separates successful prediction from this design goal.\n'
        'A prediction-supported draft remains saved if the design goal is unresolved; you can\n'
        'continue the same author account. Source support for at least one actual prospective\n'
        'sequence is required to complete this deliverable, not a number of tool calls.\n'
        'Preserve the inherited fitting methods where useful. After publication the exact same\n'
        'new program will also run with a random choice from its own equal-cost experiment pool.\n'
        'This evaluates choice rather than a different fitter. Correctness is not required to\n'
        'exceed the successful fixed-suite baseline. All comparisons retain 192 experimental rows.\n\n'
        + author.common_interface).replace('probe_case', 'design_case').replace(
            '12 program invocations', '32 program invocations')


def conditional_design_protocol(source, old, author, *, model, timeout_seconds):
    """Reconnect the historical conditional laboratory; retain every earlier target."""
    plan = prospective_design_protocol(source, old, author, model=model,
                                       timeout_seconds=timeout_seconds)
    retained_limits = {**old['limits'], 'confirmation_mse_threshold': 1e-10}
    practices = {name: {**deepcopy(case), 'limits': deepcopy(retained_limits)}
                 for name, case in old['practice_cases'].items()}
    targets = [{**deepcopy(case), 'limits': deepcopy(retained_limits)}
               for case in old['transfer_cases']]
    retained_ids = [case['id'] for case in targets]
    def case(seed, stage):
        return {'kind': 'conditional_window', 'apparatus_seed': seed, 'seed': seed,
            'stage': stage, 'noise_sigma': 0., 'prefix_seed': seed + 100003,
            'confirmation_seed': seed + 200003, 'evaluation_seed': seed + 300007,
            'limits': {'confirmation_mse_threshold': 1e-6}}
    for index, stage in enumerate((1, 2, 4)):
        practices[f'design_case_{len(practices):03d}'] = case(131011031 + index * 1009, stage)
    for block, seed in enumerate((132011031, 132021040, 132031049, 132041058)):
        for stage in (1, 2, 4):
            targets.append({'id': f'conditional_target_{len(targets)-len(retained_ids):03d}',
                            'parameter_pair': block, **case(seed, stage)})
    plan.update(study='conditional_window_experiment_method_transfer', conditional_design=True,
        practice_cases=practices, previous_practice_cases={name: name for name in old['practice_cases']},
        transfer_cases=targets, old_target_ids=retained_ids,
        limits={**old['limits'], 'observation_budget': 2240, 'compute_budget': 640000,
                'execution_seconds': 7200., 'confirmation_mse_threshold': 1e-10},
        author_limits={**old['author_limits'], 'max_model_calls': 96, 'token_budget': 20_000_000,
            'observation_budget': 120000, 'compute_budget': 8_000_000, 'execution_seconds': 36000.},
        restoration='Reuse paid source practice under identical old-case conditions; every new '
            'case requires real practice. Any source edit invalidates its earlier coverage.',
        practice_protocol='Six unchanged source practices plus three anonymous public apparatuses '
            'from the historical conditional/window laboratory. All have 72 paid prefix rows and '
            '96 blind confirmation rows. Old cases keep 360 observations, new ones allow 2240. '
            'The per-case criterion is 1e-10 for old cases and the historical 1e-6 for new cases. '
            'All allow 32 program invocations. No confirmation labels reenter a frozen program.',
        comparison='Retention: unchanged original and revised programs on ALL ten earlier targets, '
            'with the original seeds, prefix, limits and 1e-10 criterion. Selection: revised program, '
            'same-cost random selection and same-cost maximum prediction variance on twelve fresh '
            'conditional/window targets (four parameter blocks, three stages). These three arms '
            'share source, candidates, fitting and tools; only their experiment choice is intervened. '
            'The original one-control method is not an applicable new-task baseline.',
        scope='LLM adoption and revision of research methods on historical bounded apparatuses. '
            'Old ten-target retention and new twelve-target selection are separate outcomes. '
            'The author is prompted to investigate, not given the hidden gates, windows or fitting '
            'formulas. Task-value selection is available but optional. This is not a Code comparison.',
        failure_policy='Keep original 10/10 source and evidence immutable. An unpublished draft '
            'does not transfer. Retain every paid practice and unresolved scientific failure.')
    plan['runtime_snapshot'].extend(['scripts/run_conditional_discovery_study.py',
                                     'scripts/run_knowledge_lineage_study.py'])
    return plan


def attach_conditional_design_task(author, plan):
    attach_experiment_design_task(author)
    author.common_interface = author.common_interface.replace('probe_case', 'design_case').replace(
        '12 program invocations', '32 program invocations').replace(
        'A trial receives 2240 observations',
        'A trial receives its per-case allowance from practice_interfaces.json').replace(
        'at MSE <= 1e-10.', 'at its per-case confirmation_mse_threshold in research_state.json.').replace(
        'mean squared error (MSE) is at most 1e-10.',
        'mean squared error (MSE) is at most the per-case confirmation_mse_threshold.').replace(
        'Every listed design_case requires fresh paid support. Old trials remain research experience\n'
        'but were run under a different confirmation protocol and do not count as current support.',
        'Old practices with the same case, source and conditions retain their paid support.\n'
        'New cases require actual practice; any edit invalidates coverage for that source.')
    public = {name: {'public_spec': laboratory(case).public_spec(),
        'limits': {**plan['limits'], **case.get('limits', {}), 'max_model_calls': 0, 'token_budget': 0}}
        for name, case in plan['practice_cases'].items()}
    text = json.dumps(public, indent=2)
    author.files['practice_interfaces.json'] = text
    write_text_incrementally(author.workspace / 'practice_interfaces.json', text)
    author.common_interface = (
        'Extend the inherited successful procedure to the listed public apparatus interfaces.\n'
        'practice_interfaces.json gives only public controls, horizons and per-case allowances.\n'
        'Keep useful old behavior. Observe the new cases, form and test explanations, and revise\n'
        'your experiment-selection method when its paid outcomes warrant a change. The common\n'
        'compare_experiments operator also accepts reference_actions (public unobserved inputs)\n'
        'to value expected prediction-risk reduction. You decide whether and how to use it.\n'
        'No new fitted model, hidden condition, experiment menu or scientific formula is supplied.\n'
        'After publication, the same program runs on unseen apparatuses with its own selection,\n'
        'a same-cost random selection, and a same-cost maximum-variance selection. Every failed\n'
        'practice remains available for your next revision. Old successes are kept separately.\n\n'
        + author.common_interface + '\n' + laboratory(next(case for case in
            plan['practice_cases'].values() if case['kind'] == 'conditional_window')).task_interface)


def repair_protocol(source, old, repair_study, *, model):
    """Adapt the existing revision study to the two observed representation repairs."""
    repair_study = Path(repair_study).resolve()
    repairs = json.loads((repair_study / 'protocol.json').read_text())
    practice = deepcopy(old['practice_cases'])
    repair_cases = {}
    for case in repairs['cases']:
        if case['kind'] == 'finite2':
            name = f'case_{len(practice):03d}'
            practice[name] = {key: case[key] for key in (
                'kind', 'seed', 'apparatus_seed', 'prefix_seed', 'confirmation_seed')}
            repair_cases[case['id']] = name
    targets = deepcopy(old['transfer_cases'])
    for block, seed in enumerate((81231031, 81232040)):
        for kind in ('recurrent', 'finite2'):
            targets.append({'id': f'new_target_{len(targets) - len(old["transfer_cases"]):03d}',
                'parameter_pair': block, 'kind': kind, 'seed': seed, 'apparatus_seed': seed,
                'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
                'evaluation_seed': seed + 300007})
    return {'study': 'recurrent_published_method_repair_transfer', 'source': str(source),
        'repair_study': str(repair_study), 'model': model,
        'provider': {'backend': 'codex_cli', 'timeout_seconds': 600., 'reasoning_effort': 'medium'},
        'author_limits': {**old['author_limits'], 'max_model_calls': 96,
            'token_budget': 20_000_000, 'observation_budget': 36_000,
            'execution_seconds': 7200., 'compute_budget': 4_000_000},
        'limits': {**old['limits'], 'observation_budget': 1152, 'execution_seconds': 3600.},
        'prefix_rows': old['prefix_rows'], 'practice_cases': practice,
        'repair_cases': repair_cases, 'transfer_cases': targets,
        'old_target_ids': [case['id'] for case in old['transfer_cases']],
        'evaluation': deepcopy(old['evaluation']),
        'restoration': 'Restore the published source and every earlier paid practice without replay. '
            'Append every registered repair-study investigation, including failures, as background '
            'experience. Existing executions do not establish publication coverage on the added cases.',
        'author_experience': 'All eight registered repair investigations are available once in the '
            'author workspace, indexed by source_catalog.json and paid traces. Source terminal '
            'evaluations are excluded. This experience is not copied into practice or target tasks.',
        'comparison': 'Both frozen methods and all practices receive the identical original archive. '
            'Original archive paths and code remain unchanged. The repair-study files are author-only '
            'experience: only the self-contained method and its learned description transfer. '
            'The revised method must not read the author workspace or repair archives. Six old targets are '
            'revisits; four new targets cover two new parameter blocks and both apparatus families.',
        'terminal_order': 'Finish both frozen research cohorts before generating any terminal labels.',
        'failure_policy': 'One author with a fixed allowance and no live consumer LLM calls. Without '
            'publication, retain every draft and failure and leave paired target slots unavailable.',
        'scope': 'One prompted method revision from two development parameter blocks. Source terminal '
            'evaluations are excluded. New parameter transfer is within two existing apparatus families; '
            'revisits are not independent discoveries or evidence of autonomous recursive advantage.',
        'runtime_snapshot': sorted({str(path.relative_to(ROOT))
            for path in (ROOT / 'src/hegel_machine').rglob('*.py')} | set(old.get('runtime_snapshot', [])) | {
            'scripts/run_anonymous_method_revision.py', 'scripts/run_anonymous_method_learning.py',
            'scripts/run_temporal_counterexample_learning.py', 'scripts/run_anonymous_revision_choice.py'})}


def _merge_author_source_catalog(author, entries):
    """Keep author-only source metadata in its existing catalogue, once per ID."""
    catalog = deepcopy(author.knowledge.get('source_catalog', []))
    positions = {entry['id']: index for index, entry in enumerate(catalog)}
    for entry in entries:
        if entry['id'] in positions:
            catalog[positions[entry['id']]] = deepcopy(entry)
        else:
            positions[entry['id']] = len(catalog)
            catalog.append(deepcopy(entry))
    text = json.dumps(catalog, indent=2)
    write_text_incrementally(author.workspace / 'source_catalog.json', text)
    author.files['source_catalog.json'] = text
    # InquiryLLMAgent owns a deep copy. The practice closure and future targets
    # retain their original knowledge; only the author's metadata changes here.
    author.knowledge['source_catalog'] = catalog
    author.knowledge.setdefault('files', {})['source_catalog.json'] = text


def attach_repair_archive(author, repair_study):
    """Materialize author-only experience once; keep practice knowledge unchanged."""
    added, records = load_research_archive(repair_study)
    prefix = 'repair_archive/'
    for name, contents in added['files'].items():
        if name != 'source_catalog.json':
            path = author.workspace / (prefix + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
    catalog = deepcopy(added['source_catalog'])
    for entry in catalog:
        entry['id'] = 'repair_' + entry['id']
        for key in ('directory', 'model', 'paid_trace'):
            if entry.get(key) is not None:
                entry[key] = prefix + entry[key]
    for entry in records:
        entry['id'] = 'repair_' + entry['id']
        for key in ('directory', 'model', 'model_context_directory'):
            if entry.get(key) is not None:
                entry[key] = prefix + entry[key]
    _merge_author_source_catalog(author, catalog)
    return records


def attach_paid_repairs(author, plan, records):
    """Expose observed repair traces without purchasing or claiming new practice support."""
    entries = []
    for record in records:
        if record['case'] not in plan['repair_cases']:
            continue
        relative = Path('repair_evidence') / record['id']
        workspace = author.workspace / relative
        workspace.mkdir(parents=True)
        view = SimpleNamespace(workspace=workspace, files={}, common_interface='')
        supply_counterexample(view, Path(record['workspace']) / 'training.json',
                              case_label=plan['repair_cases'][record['case']])
        entries.append({'case': plan['repair_cases'][record['case']], 'source': record['id'],
            'ledger': str(relative / 'counterexample.json'),
            'trace': str(relative / 'counterexample_trace.json'),
            'observations': str(relative / 'counterexample_observations.json')})
    evidence = {'records': entries, 'costs': total_costs([row['costs'] for row in records]),
        'use': 'Historical paid research, including successful repairs and failed attempts. '
            'No new acquisition and no publication support for the added practice cases.'}
    text = json.dumps(evidence, indent=2)
    write_text_incrementally(author.workspace / 'repair_evidence.json', text)
    author.files['repair_evidence.json'] = text
    task = ('Revise the inherited research program using the supplied paid research experience.\n'
        'source_catalog.json retains all original and repair-study investigations, including failures.\n'
        'repair_evidence.json locates the operation ledgers, full traces and paid observations for\n'
        'the added practice cases. Paths inside each ledger are relative to its own directory.\n'
        'Earlier supported drafts and practices remain in method_candidates.json and resume_context.json.\n'
        'Choose what to retain or revise from evidence. Preserve useful earlier behavior and establish\n'
        'support for one final source on every listed case through paid practice and finish.\n'
        'The added repair_archive and repair_evidence directories are author-only experience.\n'
        'Every practice and target receives the original archive, not these additional source files.\n'
        'Publish a self-contained method: no reads of the author workspace, repair archives or other\n'
        'task workspaces. Only the method and original archive transfer; practice-fitted files do not.\n\n')
    author.common_interface = task + 'AUTHOR ACTIONS:' + author.common_interface.split('AUTHOR ACTIONS:', 1)[1]
    author.common_interface = author.common_interface.replace(
        'The initial method is only an unresolved placeholder. You author its decisions,\n'
        'predictors, parameter estimation and experiments using ordinary Python.',
        'The inherited source and saved drafts are revisable Python programs with recorded paid support.')
    return evidence


def attach_online_research_experience(author, experience):
    """Index successful paid online research for compilation, without transferring its fit."""
    source = Path(experience).resolve()
    result = json.loads((source / 'training.json').read_text())
    if (result['status'] != 'completed' or not result.get('model')
            or not (result.get('fresh_confirmation') or {}).get('passed')):
        raise ValueError('method compilation requires an independently confirmed online result')
    view = author.workspace / 'online_research_experience'
    view.mkdir(exist_ok=True)
    names = {'shared', 'training.json', 'archive.json', 'laboratory_observations.json',
        'research_state.json', 'research_context.json', 'research_trace.json', 'method_decision.json'}
    names.add(Path(result['model']).parts[0])
    available = []
    for name in sorted(names):
        origin = source / name
        if origin.exists():
            target = view / name
            if not target.exists():
                target.symlink_to(origin, target_is_directory=origin.is_dir())
            available.append(str(Path(view.name) / name))
    evidence = {'training': str(Path(view.name) / 'training.json'),
        'model': str(Path(view.name) / result['model']), 'files': available,
        'costs': result['costs'],
        'fresh_confirmation': {key: result['fresh_confirmation'][key]
            for key in ('passed', 'mse', 'rows', 'mse_threshold') if key in result['fresh_confirmation']},
        'fresh_confirmation_source': {'file': str(Path(view.name) / 'training.json'),
                                      'json_path': ['fresh_confirmation']},
        'use': 'Previously paid online research for method compilation. Read these references only; '
            'all fitting, observations and unsuccessful hypotheses belong to the source task. '
            'This supplies no practice support. Only the newly published self-contained method '
            'and original archive transfer; neither this fitted model nor its observations transfer.'}
    # The source workspace contains failed old procedures as well as a confirmed
    # predictor. Keep their existing execution facts together, rather than making
    # the author infer publication from the shared directory's location.
    execution = {'working_directory': view.name,
        'path_scope': 'Paths and JSON references in this section are relative to working_directory.',
        'trace_fields': {'online_actions': 'training.json.events',
                         'procedure_steps': 'training.json.method_results'},
        'confirmation_scope': 'The source confirmation supports its submitted predictor; '
                              'it does not publish a research procedure.'}
    decision_path = source / 'method_decision.json'
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text())
        execution['method_decision_file'] = 'method_decision.json'
        execution['methods'] = []
        for method in decision.get('methods', []):
            # current_executions is a display projection and may contain saved
            # references. Summarize the already-loaded canonical executions,
            # preserving their global method_results indices and full feedback.
            summary = practice_execution_summary(result, '.', method_path=method['path'])
            continuation = (method.get('continuation') or {}).get('status')
            if continuation is not None:
                summary['status'] = continuation
            summary.update(result_file='training.json', result_steps_field='method_results')
            summary.pop('author_feedback_steps_field', None)
            # This card already links the whole source procedure history and
            # separately previews recent online actions; do not repeat all steps.
            summary.pop('research_steps', None)
            execution['methods'].append({
                **{key: method[key] for key in ('id', 'path', 'source_unchanged') if key in method},
                'prior_execution': {**{key: method.get('prior_execution', {})[key]
                    for key in ('scope',) if key in method.get('prior_execution', {})},
                    'cases': [{key: case[key] for key in ('case', 'status', 'fresh_confirmation')
                               if key in case}
                              for case in method.get('prior_execution', {}).get('cases', [])]},
                'execution_summary': summary})
        execution['native_research'] = []
        for index, option in enumerate(decision.get('research_options', [])):
            if option.get('id') != 'native_condition_research':
                continue
            rounds = option.get('current_executions', [])
            execution['native_research'].append({'id': option['id'],
                'execution_count': len(rounds),
                'source': {'file': 'method_decision.json', 'json_path': ['research_options', index]},
                'recent_executions': [{
                    **{key: item[key] for key in ('question_id', 'round_index', 'status', 'paid_rows', 'source')
                       if key in item},
                    'model_count': len(item.get('models', [])),
                    'model_preview': [{key: model[key] for key in ('model', 'fit_mse', 'evaluate_action')
                                       if key in model} for model in item.get('models', [])[:3]]}
                    for item in rounds[-3:]]})
        # Executable paths in the source card need the same read-only view as
        # its state references. Exported candidates live outside shared/.
        model_roots = {Path(model['model']).parts[0]
            for research in execution['native_research']
            for item in research['recent_executions']
            for model in item['model_preview'] if model.get('model')}
        for name in sorted(model_roots - names):
            origin = source / name
            if origin.exists():
                target = view / name
                if not target.exists():
                    target.symlink_to(origin, target_is_directory=origin.is_dir())
                available.append(str(Path(view.name) / name))
    events = result.get('events', [])
    start = max(0, len(events) - 12)
    execution.update(online_event_count=len(events), online_preview_start=start, recent_online_actions=[])
    for index, event in enumerate(events[start:], start=start):
        arguments, feedback = event.get('arguments') or {}, event.get('feedback') or {}
        action = {'event_index': index, 'step': event.get('step'), 'action': event.get('action'),
            'arguments': {key: arguments[key] for key in ('operator', 'model', 'path') if key in arguments},
            'feedback': {key: feedback[key] for key in ('status', 'mse', 'returncode', 'timeout')
                         if key in feedback},
            'source': {'file': 'training.json', 'json_path': ['events', index]}}
        if feedback.get('error'):
            action['feedback']['error'] = str(feedback['error'])[:1200]
        if isinstance(feedback.get('fresh_confirmation'), dict):
            confirmation = feedback['fresh_confirmation']
            action['feedback']['fresh_confirmation'] = {key: confirmation[key]
                for key in ('passed', 'mse', 'rows', 'mse_threshold') if key in confirmation}
        execution['recent_online_actions'].append(action)
    evidence['source_execution'] = execution
    submitted = result['fresh_confirmation'].get('submitted_candidate')
    if submitted:
        # Keep this compact predictor context discoverable with ordinary rglob.
        # A directory symlink silently hid its source from the author's actual
        # inspection, which then read an unrelated archived predictor instead.
        # Reuse the execution-context copier: fitting inputs remain together,
        # while research histories stay in the read-only links above.
        replay = view / 'confirmed_candidate' / '_evaluated_model'
        model = _copy_model_context(source, submitted['model'], replay)
        relative = replay.relative_to(author.workspace)
        evidence['confirmed_candidate'] = {**deepcopy(submitted),
            'model': str(relative / model), 'working_directory': str(relative)}
        available.append(str(relative))
    selected = evidence.get('confirmed_candidate', {})
    model = selected.get('model', evidence['model'])
    directory = selected.get('working_directory', str(Path(model).parent))
    definitions = {node.name for node in ast.parse((author.workspace / model).read_text()).body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    _merge_author_source_catalog(author, [{
        'id': 'online_confirmed_source', 'directory': directory, 'model': model,
        'paid_trace': evidence['training'], 'status': result['status'],
        'fresh_confirmation': deepcopy(evidence['fresh_confirmation']),
        'entry_points': [name for name in ('fit', 'fit_features', 'predict', 'predict_features',
                                         'bind_models') if name in definitions],
        'scope': 'Read-only author experience, absent from practice and target input archives.',
        'program_reuse': 'The source program can be inspected and reused with new-task fitting. '
            'Its saved coefficients, fitted state and observations belong to this source; '
            'new task support requires fitting and testing on that task\'s paid evidence.'}])
    evidence['source_catalog_entry'] = 'online_confirmed_source'
    text = json.dumps(evidence, separators=(',', ':'))
    write_text_incrementally(author.workspace / 'online_research_experience.json', text)
    author.files['online_research_experience.json'] = text
    author.common_interface = (
        'online_research_experience.json indexes an independently confirmed online investigation. '
        'Inspect its paid actions, revisions, observations and final source through run. '
        'Its confirmed_candidate, when present, references the submitted predictor together with '
        'its saved fitting rows and working directory for replay. '
        'source_execution separates the old procedure and its recorded support from ordinary '
        'online actions, including their actual evaluation and confirmation outcomes. Its paths '
        'are relative to the stated source working_directory. Predictor confirmation does not '
        'establish publication or current support for a shared research procedure. '
        'Its references are read-only author experience. Generalize a self-contained research '
        'procedure and establish its support on every practice case using the existing publication '
        'rules. No source-task fitted model or observations will enter practices or targets.\n\n'
        + author.common_interface)
    return evidence


def reproduce_counterexample(author):
    """Execute the exact earlier failing condition and record it on the author account."""
    case = author.cases[COUNTEREXAMPLE_CASE]
    source = (author.workspace / METHOD_PATH).read_text()
    directory = f'trials/trial_{len(author.practice_trials):03d}'
    limits = replace(author.practice_limits, max_model_calls=0, token_budget=0)
    payload = author.prepare_trial(case, author.workspace / directory, source, limits)
    result = payload['agent'].execute_method(payload['laboratory'], max_steps=12)
    selected_prior = bool(result.get('model') and any(
        python_model_context_matches(payload['agent'].workspace, result['model'],
            candidate['snapshot_workspace'], candidate['inner_model'])
        for candidate in payload['inherited_candidates']))
    feedback = author._record_practice(COUNTEREXAMPLE_CASE, directory, source,
        payload['agent'], result, selected_prior)
    author.events.append({'step': -1, 'action': 'run',
        'arguments': {'operator': 'practice', 'case': COUNTEREXAMPLE_CASE},
        'feedback': deepcopy(feedback),
        'usage': {'backend': 'development_counterexample_reproduction', 'total_tokens': 0}})
    return {**feedback, 'effective_confirmation_seed': case['confirmation_seed']}


def attach_revision_task(author, result):
    """Reuse the historical counterexample view; the author chooses the repair."""
    task = (
        'Revise the existing research program using its paid counterexample.\n'
        'The current draft may be unsupported. Earlier sources and every paid practice\n'
        'remain available in method_candidates.json and resume_context.json.\n'
        'counterexample.json identifies the actual rejected execution on ' + COUNTEREXAMPLE_CASE +
        ',\nits executed source, proposals and feedback. Inspect that evidence and decide\n'
        'which source to retain or revise, preserving behavior still supported by the\n'
        'other supplied cases. Use practice and finish to establish support for the\n'
        'final source on every listed case. Only the published self-contained method\n'
        'and original archive transfer. The earlier archive is background evidence.\n\n')
    # Adapt the old reopening task, keeping the existing author actions, paid
    # practice and publication requirements rather than replacing their runtime.
    author.common_interface = task + 'AUTHOR ACTIONS:' + author.common_interface.split('AUTHOR ACTIONS:', 1)[1]
    author.common_interface = author.common_interface.replace(
        'The initial method is only an unresolved placeholder. You author its decisions,\n'
        'predictors, parameter estimation and experiments using ordinary Python.',
        'The inherited method and saved drafts are revisable Python programs.\n'
        'Their paid outcomes identify existing support and unresolved failures.')
    return supply_counterexample(author, result, case_label=COUNTEREXAMPLE_CASE)


def run_revision(plan, output, *, provider_factory):
    output, source = Path(output).resolve(), Path(plan['source'])
    if (output / 'protocol.json').exists():
        raise ValueError('use a fresh output directory; retain earlier author attempts')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', plan)
    for name in plan.get('runtime_snapshot', []):
        destination = output / 'source_at_run' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    previous = json.loads((source / 'author' / 'training.json').read_text())
    original_knowledge = json.loads((source / 'initial_archive.json').read_text())
    sources = json.loads((source / 'source_outcomes.json').read_text())
    knowledge = register_published_method(original_knowledge, previous, max_steps=plan.get('method_steps', 12))
    write_json(output / 'initial_archive.json', knowledge, base=source / 'initial_archive.json')
    write_json(output / 'source_outcomes.json', sources)
    reference = output / 'reference_archive'
    previous_reference = source / 'reference_archive'
    paid_traces = archived_paid_trace_references(original_knowledge['files'], reference,
        source_workspace=previous_reference if previous_reference.is_dir() else None)
    for name, contents in original_knowledge['files'].items():
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in paid_traces:
            if path.resolve() != paid_traces[name]:
                path.symlink_to(paid_traces[name])
        elif ((previous_reference / name).is_file()
                and (previous_reference / name).read_text(encoding='utf-8') == contents):
            _copy_model_file(previous_reference / name, path)
        else:
            write_text_incrementally(path, contents)
    workspace = output / 'author'
    practice_options = {}
    learner_type = AnonymousResearchMethodLearner
    if plan.get('prospective_design'):
        from scripts.experimental_design_author import ExperimentalDesignMethodLearner
        from scripts.experimental_design_control import (
            ExperimentalDesignMethodAgent, RandomExperimentMethodAgent, VarianceExperimentMethodAgent)
        learner_type = ExperimentalDesignMethodLearner
        practice_options['agent_type'] = ExperimentalDesignMethodAgent
    elif plan.get('design_experiments'):
        from hegel_machine.cognition.anonymous_method_learning import SingleSubmissionMethodAgent
        practice_options['agent_type'] = SingleSubmissionMethodAgent
    author = learner_type(provider_factory(workspace), workspace,
        knowledge=knowledge, paid_trace_references=paid_traces, cases=plan['practice_cases'], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan['author_limits']),
        practice_limits=InquiryLLMConfig(**plan['limits']),
        prepare_trial=trial_preparer(knowledge, sources, reference, **practice_options))
    restored = previous
    if plan.get('continued_author'):
        path = Path(plan['continued_author'])
        restored = json.loads(path.read_text())
        if not plan.get('design_experiments'):
            baseline = json.loads((path.parent.parent / 'baseline_reproduction.json').read_text())
    author.restore_practice(restored)
    if plan.get('design_experiments'):
        if plan.get('conditional_design'):
            attach_conditional_design_task(author, plan)
        elif plan.get('prospective_design'):
            attach_prospective_design_task(author)
        else:
            attach_experiment_design_task(author)
        baseline = {'costs': previous['costs'], 'scope': 'Previously paid method-author experience; '
            'retained in prior_author_phases and excluded from new costs.'}
    elif plan.get('repair_study'):
        repair_records = attach_repair_archive(author, plan['repair_study'])
        baseline = attach_paid_repairs(author, plan, repair_records)
        write_json(output / 'repair_evidence.json', baseline)
        write_json(output / 'author_experience.json', {'source': plan['repair_study'],
            'records': repair_records, 'scope': plan['author_experience'],
            'transfer_knowledge': 'Unchanged original archive; additional experience is author-only.'})
    else:
        if not plan.get('continued_author'):
            baseline = reproduce_counterexample(author)
        counterexample = json.loads((workspace / baseline['directory'] / 'result.json').read_text())
        attach_revision_task(author, counterexample)
        write_json(output / 'baseline_reproduction.json', baseline)
    if plan.get('online_research_experience'):
        attach_online_research_experience(author, plan['online_research_experience'])
    write_json(output / 'revision_context.json', {
        'baseline': baseline, 'restored_practice_trials': len(restored['practice_trials'])})
    write_json(workspace / 'active.json', author.result('researching'))
    learned = author.run(on_checkpoint=lambda data: write_json(workspace / 'active.json', data))
    write_json(workspace / 'training.json', learned)
    published = learned['status'] == 'completed' and learned.get('method_source') is not None
    write_json(output / 'published_method.json', {'published': published,
        'source': learned.get('method_source'), 'lesson': learned.get('lesson'),
        'practice_trials': learned['practice_trials']})
    return run_revision_targets(plan, output, previous, learned, original_knowledge, sources)


def run_revision_targets(plan, output, previous, learned, original_knowledge, sources):
    """Finish frozen cohorts from a published author without any live provider."""
    output, source = Path(output).resolve(), Path(plan['source'])
    old_plan = json.loads((source / 'protocol.json').read_text())
    context = json.loads((output / 'revision_context.json').read_text())
    baseline = context['baseline']
    published = learned['status'] == 'completed' and learned.get('method_source') is not None
    if plan.get('prospective_design'):
        from scripts.experimental_design_control import (
            ExperimentalDesignMethodAgent, RandomExperimentMethodAgent, VarianceExperimentMethodAgent)
    elif plan.get('design_experiments'):
        from hegel_machine.cognition.anonymous_method_learning import SingleSubmissionMethodAgent
    frozen = {'original': [], 'revised': []}
    if plan.get('prospective_design'):
        frozen['random_choice'] = []
    if plan.get('conditional_design'):
        frozen['variance_choice'] = []
    if published:
        publications = [('original', previous), ('revised', learned)]
        if plan.get('prospective_design'):
            publications.append(('random_choice', learned))
        if plan.get('conditional_design'):
            publications.append(('variance_choice', learned))
        for name, publication in publications:
            memory = register_published_method(original_knowledge, publication,
                                               max_steps=plan.get('method_steps', 12))
            branch = output / name
            branch.mkdir(exist_ok=True)
            write_json(branch / 'transfer_archive.json', memory)
            if plan.get('prospective_design'):
                target_plan = plan
                if plan.get('conditional_design') and name != 'revised':
                    target_plan = {**plan, 'transfer_cases': [case for case in plan['transfer_cases']
                        if (case['id'] in plan['old_target_ids']) == (name == 'original')]}
                controls = {'random_choice': RandomExperimentMethodAgent,
                            'variance_choice': VarianceExperimentMethodAgent}
                frozen[name] = frozen_transfers(target_plan, branch, memory, sources,
                    publication['method_source'], agent_type=controls.get(name, ExperimentalDesignMethodAgent))
            elif plan.get('design_experiments'):
                old_ids = set(plan['old_target_ids'])
                retained_plan = {**plan, 'transfer_cases': [case for case in plan['transfer_cases']
                    if case['id'] in old_ids]}
                probe_plan = {**plan, 'transfer_cases': [case for case in plan['transfer_cases']
                    if case['id'] not in old_ids]}
                frozen[name] = frozen_transfers(retained_plan, branch, memory, sources,
                    publication['method_source'])
                frozen[name] += frozen_transfers(probe_plan, branch, memory, sources,
                    publication['method_source'], agent_type=SingleSubmissionMethodAgent)
            else:
                frozen[name] = frozen_transfers(plan, branch, memory, sources, publication['method_source'])
    # No earlier scoring: both frozen research cohorts have now stopped.
    records = {name: score_frozen(plan, results) for name, results in frozen.items()}
    if plan.get('design_experiments'):
        from scripts.run_anonymous_online_research import experimental_design_trace
        for name, results in frozen.items():
            for record, (_, result) in zip(records[name], results):
                record['experimental_design'] = experimental_design_trace(result)
                record['research_protocol'] = ('original_confirmation_feedback'
                    if record['case'] in plan['old_target_ids'] and not plan.get('prospective_design')
                    else 'single_submission')
    pairs = []
    revised_by_case = {row['case']: row for row in records['revised']}
    for old in records['original']:
        new = revised_by_case[old['case']]
        pairs.append({'case': old['case'], 'kind': old['kind'],
            'group': ('old_target' if old['case'] in plan['old_target_ids'] else
                      'new_parameter' if plan.get('repair_study') or plan.get('design_experiments')
                      else 'new_object_prefix'),
            'original_passed': old['terminal']['passed'], 'revised_passed': new['terminal']['passed'],
            'original_status': old['status'], 'revised_status': new['status'],
            'observation_delta': new['costs']['observation_cost'] - old['costs']['observation_cost'],
            'compute_delta': new['costs']['compute_cost'] - old['costs']['compute_cost']})
    target_costs = {name: total_costs([result['costs'] for _, result in results])
                    for name, results in frozen.items()}
    earlier_phases = plan.get('prior_author_phases') or [*old_plan.get('prior_author_phases', []), {
        'result': str(source / 'author' / 'training.json'), 'status': previous['status'],
        'costs': previous['costs'], 'usage': usage_summary(previous)}]
    summary = {'study': plan['study'], 'status': 'completed', 'method_published': published,
        'source_changed': published and learned['method_source'] != previous['method_source'],
        'author_status': learned['status'], 'author_usage': usage_summary(learned),
        'prior_author_phases': earlier_phases,
        'old_author_costs': total_costs([phase['costs'] for phase in earlier_phases]),
        'baseline_costs': baseline['costs'], 'author_costs': learned['costs'],
        'baseline_cost_scope': ('Historical repair-study costs, excluded from this author and new costs.'
            if plan.get('repair_study') else
            'Already included in prior_author_phases; no new acquisition or charge.'
            if plan.get('continued_author') or plan.get('design_experiments') else
            'Included in author_costs and new_costs exactly once.'),
        'baseline_reused': bool(plan.get('continued_author') or plan.get('repair_study')
                               or plan.get('design_experiments')),
        'source_research_costs': total_costs([row['costs'] for row in sources]),
        'target_costs': target_costs,
        'new_costs': total_costs([learned['costs'], *target_costs.values()]),
        'restored_practice_trials': context['restored_practice_trials'],
        'new_practice_trials': len(learned['practice_trials']) - context['restored_practice_trials'],
        'frozen': records, 'pairs': pairs,
        'unavailable_target_episodes': 0 if published else (
            2 * len(plan['old_target_ids']) + 3 * (len(plan['transfer_cases']) - len(plan['old_target_ids']))
            if plan.get('conditional_design') else len(frozen) * len(plan['transfer_cases'])),
        'live_consumer_model_calls': 0, 'scope': plan['scope']}
    if plan.get('prospective_design'):
        summary['design_objective'] = learned.get('design_objective')
        summary['predictive_publication'] = learned.get('predictive_publication')
        summary['choice_pairs'] = [{'case': active['case'],
            'selected_passed': active['terminal']['passed'],
            'random_passed': control['terminal']['passed'],
            'observation_delta': active['costs']['observation_cost'] - control['costs']['observation_cost'],
            'compute_delta': active['costs']['compute_cost'] - control['costs']['compute_cost']}
            for control in records['random_choice'] for active in [revised_by_case[control['case']]]]
    if plan.get('conditional_design'):
        summary['variance_pairs'] = [{'case': control['case'],
            'selected_passed': revised_by_case[control['case']]['terminal']['passed'],
            'variance_passed': control['terminal']['passed'],
            'observation_delta': revised_by_case[control['case']]['costs']['observation_cost']
                - control['costs']['observation_cost'],
            'compute_delta': revised_by_case[control['case']]['costs']['compute_cost']
                - control['costs']['compute_cost']} for control in records['variance_choice']]
    write_json(output / 'summary.json', summary)
    return summary


def resume_revision_targets(study):
    """Reuse frozen publication and completed target results from this exact study."""
    output = Path(study).resolve()
    plan = json.loads((output / 'protocol.json').read_text())
    learned = json.loads((output / 'author' / 'training.json').read_text())
    publication = json.loads((output / 'published_method.json').read_text())
    if (not publication.get('published') or learned['status'] != 'completed'
            or not learned.get('method_source')
            or publication.get('source') != learned['method_source']):
        raise ValueError('target continuation requires the actual published source and completed author')
    source = Path(plan['source'])
    previous = json.loads((source / 'author' / 'training.json').read_text())
    knowledge = json.loads((source / 'initial_archive.json').read_text())
    sources = json.loads((source / 'source_outcomes.json').read_text())
    return run_revision_targets(plan, output, previous, learned, knowledge, sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume-targets', type=Path,
        help='Resume only missing frozen targets from an actually published study; no author calls')
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--timeout-seconds', type=float, default=300.)
    parser.add_argument('--continue-author', type=Path,
        help='Restore a real unpublished author phase and its paid counterexample without replay')
    parser.add_argument('--repair-study', type=Path,
        help='Append the completed eight-episode revision-choice study without replaying its evidence')
    parser.add_argument('--design-experiments', action='store_true',
        help='Revise the published method with new single-submission experiment-design practice')
    parser.add_argument('--prospective-design', action='store_true',
        help='Learn prospective experiment design and compare choices within the same program')
    parser.add_argument('--conditional-design', action='store_true',
        help='Reuse historical conditional/window tasks while retaining all earlier targets')
    args = parser.parse_args()
    if args.resume_targets:
        summary = resume_revision_targets(args.resume_targets)
        print(json.dumps({'method_published': summary['method_published'],
            'source_changed': summary['source_changed'], 'new_costs': summary['new_costs']}), flush=True)
        return
    if args.source is None or args.output is None:
        parser.error('--source and --output are required unless --resume-targets is selected')
    plan = protocol(args.source, model=args.model, timeout_seconds=args.timeout_seconds,
                    continue_author=args.continue_author, repair_study=args.repair_study,
                    design_experiments=args.design_experiments, prospective_design=args.prospective_design,
                    conditional_design=args.conditional_design)
    plan['provider']['auth_mode'] = verify_codex_subscription()
    def provider(workspace):
        return CodexCodeProvider(workspace, model=args.model, reasoning_effort='medium',
            timeout_seconds=plan['provider']['timeout_seconds'])
    summary = run_revision(plan, args.output, provider_factory=provider)
    print(json.dumps({'method_published': summary['method_published'],
        'source_changed': summary['source_changed'], 'new_costs': summary['new_costs']}), flush=True)


if __name__ == '__main__':
    main()
