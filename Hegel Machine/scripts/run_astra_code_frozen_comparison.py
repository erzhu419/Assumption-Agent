#!/usr/bin/env python3
"""Let ordinary Astra Code compile its own source research, then freeze 22 targets."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_method_learning import AUTHOR_INTERFACE
from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_method_learning import PythonArchiveMethodLearner
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, register_published_method
from scripts.experimental_design_control import ExperimentalDesignMethodAgent
from scripts.run_anonymous_method_learning import frozen_transfers, laboratory, score_frozen, trial_preparer
from scripts.run_anonymous_method_revision import attach_online_research_experience
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_python_method_learning_study import usage_summary


CODE_AUTHOR_TASK = """Compile an ordinary Python program that solves every listed apparatus.
Use the inherited code, paid practice records and your own confirmed source investigation.
Choose algorithms, fitting procedures and experiments yourself. The objective is accurate
predictions on fresh trajectories. You may refit, reuse or replace earlier predictors.

AUTHOR ACTIONS:
write {path:'shared/consolidated_method.py',code:'Python source'};
run {code:'Python source'} to inspect files or edit your program;
run {operator:'practice',case:'design_case_000'} to execute your current program;
run {operator:'restore_method',trial:'trials/trial_NNN'} to restore a paid draft;
finish {path:'shared/consolidated_method.py',lesson:'...'} to request publication.
Writing or editing does not automatically run a practice. Explicit practice and finish
spend the remaining author account. Finish practices missing cases in listed order;
the first failure returns its evidence. The latest supported old draft may also be
compared using the same account. method_candidates.json and method_decision.json
describe the available paid drafts and their exact-source support.

Publication requires this exact source to pass fresh confirmation on all listed cases.
Each practice starts with the original archive and a paid 72-row prefix. It allows
32 program invocations without an LLM. The FIRST model submission ends research;
96 fresh confirmation rows then test it. A failed confirmation ends that practice.
The author may inspect this feedback and change its next draft. An edit needs support
for its own source. practice_interfaces.json lists public controls and per-case limits.
Only the frozen program and original archive transfer to the 22 targets; source and
practice fitted models, data and target results do not transfer. Target execution
uses the same first-submission contract, public tools and physical limits, with no LLM.
"""


def read_json(path):
    return json.loads(Path(path).read_text())


class CodeProgramLearner(PythonArchiveMethodLearner):
    """Ordinary explicit coding/practice; predictive coverage is its publication task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # PairedMethodLearner uses InquiryLLMAgent, which creates no automatic
        # CommonArchive research root. Select Code's ordinary result/action policy.
        self.arm = 'code_react'
        contract = 'PROGRAM INPUTS:' + AUTHOR_INTERFACE.split('PROGRAM INPUTS:', 1)[1]
        contract = contract.replace('at MSE <= 1e-10.',
            'at the per-case confirmation_mse_threshold in research_state.json.')
        contract = contract.replace('Failed confirmation becomes paid evidence on the next invocation.',
            'Confirmation ends this practice; its feedback is available to the author.')
        contract = contract.replace('Failed confirmation contributes paid counterexamples, visible on the next invocation.',
            'Confirmation ends this practice; its counterexamples are available to the author.')
        self.common_interface = CODE_AUTHOR_TASK + contract

    def _prior_selection_error(self, coverage):
        return None

    def result(self, status):
        result = super().result(status)
        # Code normally exports no knowledge. Preserve this single editable draft
        # so the existing paid author restore can resume an interrupted compilation.
        result['outgoing_knowledge']['files'] = {
            METHOD_PATH: (self.workspace / METHOD_PATH).read_text()}
        return result


class FrozenCodeProgramAgent(ExperimentalDesignMethodAgent):
    """Same frozen tools/32-step submission contract, without an automatic Hegel root."""

    def __init__(self, *args, **kwargs):
        kwargs['arm'] = 'code_react'
        super().__init__(*args, **kwargs)


def protocol(baseline, code_source):
    baseline, code_source = Path(baseline).resolve(), Path(code_source).resolve()
    # The root protocol predates Astra's successful Hegel research. method/protocol
    # contains that arm's newly attached experience and must not seed this control.
    method = read_json(baseline / 'protocol.json')['method']
    plan = {key: deepcopy(method[key]) for key in (
        'source', 'continued_author', 'model', 'provider', 'author_limits', 'limits',
        'prefix_rows', 'practice_cases', 'transfer_cases', 'evaluation', 'method_steps',
        'prior_author_phases')}
    plan.update(study='astra_code_compile_then_frozen_transfer', baseline=str(baseline),
        online_research_experience=str(code_source),
        publication='Fresh predictive confirmation on all nine cases with exactly one source.',
        terminal_order='Score only after all 22 frozen investigations stop.',
        scope='Compiled-Code capability control in the common execution and paid-practice harness. '
            'Same pre-Astra K0 and 99 author practices; only Code own confirmed source is added. '
            'Ordinary explicit coding/practice, without mandatory unchanged-prior reuse or a '
            'prospective-design publication gate. Practices and targets use code_react without '
            'automatic Hegel root focus. This is not an ablation of every Hegel component.',
        failure_policy='Retain completed failures and paid drafts; never publish an unsupported '
            'source or silently repeat an interrupted author call.')
    return plan


def prepare_author(plan, output, provider_factory):
    output, source = Path(output), Path(plan['source'])
    own_source = read_json(Path(plan['online_research_experience']) / 'training.json')
    own_terminal = read_json(Path(plan['online_research_experience']) / 'evaluation.json')
    if (own_source.get('agent_arm') != 'code_react' or own_source['status'] != 'completed'
            or not own_source.get('model')
            or not (own_source.get('fresh_confirmation') or {}).get('passed')
            or not own_terminal.get('passed')):
        raise ValueError('Compile only the independently confirmed and terminally passed Code source')
    original = read_json(source / 'initial_archive.json')
    sources = read_json(source / 'source_outcomes.json')
    earlier = read_json(source / 'author/training.json')
    knowledge = register_published_method(original, earlier, max_steps=plan['method_steps'])
    write_json(output / 'initial_archive.json', knowledge)
    write_json(output / 'source_outcomes.json', sources)
    reference = output / 'reference_archive'
    for name, contents in original['files'].items():
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    workspace = output / 'author'
    author = CodeProgramLearner(provider_factory(workspace), workspace,
        knowledge=knowledge, cases=plan['practice_cases'], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan['author_limits']),
        practice_limits=InquiryLLMConfig(**plan['limits']),
        prepare_trial=trial_preparer(knowledge, sources, reference,
                                    agent_type=FrozenCodeProgramAgent))
    restored = read_json(plan['continued_author'])
    author.restore_practice(restored)
    public = {name: {'public_spec': laboratory(case).public_spec(),
        'limits': {**plan['limits'], **case.get('limits', {}),
                   'max_model_calls': 0, 'token_budget': 0}}
        for name, case in plan['practice_cases'].items()}
    author.files['practice_interfaces.json'] = json.dumps(public, indent=2)
    (workspace / 'practice_interfaces.json').write_text(author.files['practice_interfaces.json'])
    author.common_interface += '\n'.join(dict.fromkeys(
        laboratory(case).task_interface for case in plan['practice_cases'].values()))
    attach_online_research_experience(author, plan['online_research_experience'])
    write_json(output / 'author_origin.json', {
        'restored_practice_trials': len(restored['practice_trials']),
        'continued_author': plan['continued_author'],
        'online_research_experience': plan['online_research_experience'],
        'source_additional_costs': own_source['continuation_cost'],
        'source_cumulative_costs': own_source['costs'],
        'prior_author_phases': plan['prior_author_phases']})
    return author, original, sources


def run_comparison(plan, output, *, provider_factory):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = output / 'protocol.json'
    if saved.exists() and read_json(saved) != plan:
        raise ValueError('Resume requires the same registered compiled-Code protocol')
    write_json(saved, plan)
    completed = output / 'summary.json'
    if completed.exists():
        return read_json(completed)
    training = output / 'author/training.json'
    if training.exists():
        learned = read_json(training)
        original = read_json(Path(plan['source']) / 'initial_archive.json')
        sources = read_json(output / 'source_outcomes.json')
    else:
        if (output / 'author').exists():
            raise ValueError('Interrupted paid author requires checkpoint recovery; preserve this workspace')
        author, original, sources = prepare_author(plan, output, provider_factory)
        write_json(output / 'progress.json', {'stage': 'ordinary_code_compilation'})
        write_json(author.workspace / 'active.json', author.result('researching'))
        learned = author.run(on_checkpoint=lambda data: write_json(author.workspace / 'active.json', data))
        write_json(training, learned)
    published = learned['status'] == 'completed' and bool(learned.get('method_source'))
    write_json(output / 'published_method.json', {'published': published,
        'source': learned.get('method_source'), 'lesson': learned.get('lesson'),
        'practice_trials': learned['practice_trials']})
    results = []
    if published:
        memory = register_published_method(original, learned, max_steps=plan['method_steps'])
        write_json(output / 'transfer_archive.json', memory)
        write_json(output / 'progress.json', {'stage': 'frozen_targets',
                                             'registered': len(plan['transfer_cases'])})
        results = frozen_transfers(plan, output, memory, sources, learned['method_source'],
                                   agent_type=FrozenCodeProgramAgent)
    write_json(output / 'progress.json', {'stage': 'terminal_scoring', 'completed': len(results)})
    records = score_frozen(plan, results)
    summary = {'study': plan['study'], 'status': 'completed' if published else 'method_unpublished',
        'method_published': published, 'author_status': learned['status'],
        'author_costs': learned['costs'], 'author_usage': usage_summary(learned),
        'origin': read_json(output / 'author_origin.json'),
        'targets': records, 'target_count': len(records),
        'target_confirmation_passed': sum(row['fresh_confirmation_passed'] for row in records),
        'target_terminal_passed': sum(row['terminal']['passed'] for row in records),
        'target_model_calls': sum(row['costs']['model_calls'] for row in records),
        'unavailable_targets': len(plan['transfer_cases']) - len(records), 'scope': plan['scope']}
    write_json(completed, summary)
    write_json(output / 'progress.json', {'stage': summary['status'],
        'completed': len(records), 'terminal_passed': summary['target_terminal_passed']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--code-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = protocol(args.baseline, args.code_source)
    cfg = plan['provider']
    cfg['auth_mode'] = verify_codex_subscription()
    provider = lambda workspace: CodexCodeProvider(workspace, model=plan['model'],
        reasoning_effort=cfg['reasoning_effort'], timeout_seconds=cfg['timeout_seconds'])
    result = run_comparison(plan, args.output, provider_factory=provider)
    print(json.dumps({'targets': result['target_count'],
                      'passed': result['target_terminal_passed']}), flush=True)


if __name__ == '__main__':
    main()
