"""Resume the existing author with actual hypothesis-testing execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.world_reference_response import (
    compare_response_proposal, return_response_to_author,
    response_premise_design_action, response_premise_proposal_inputs)
from hegel_machine.cognition.world_llm_inquiry import refresh_condition_proposal
from scripts.run_response_ambiguity_continuation import restore_saved
from scripts.run_response_family_recovery import account
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json

SOURCE = ROOT / 'artifacts/response_condition_synthesis_20261009/parent/result.json'
BASE = Path(__file__).parent


def ordinary_author_handoff(agent, lab, params, output):
    """Give the existing ordinary loop the same paid author memory and entry refresh."""
    author = agent.llm_inquiries['proposal:' + params['path']]
    previous = author['proposal']['comparison_result']
    current = compare_response_proposal(agent, lab, {
        'path': params['response_path'], 'binding_path': params['path'],
        'question': params['question']}, previous)
    return_response_to_author(agent, current)
    conditions = refresh_condition_proposal(agent, params['condition_proposal'])
    author['events'].append({'action': 'condition_return', 'arguments': {},
        'feedback': {'condition_research': conditions}})
    qid = params['question']['id']
    ordinary = agent.llm_inquiries[qid]
    write_json(output / 'session_handoff.json', {
        'source_session': 'proposal:' + params['path'], 'target_session': qid,
        'previous_target_notes': ordinary['notes'],
        'previous_target_event_count': len(ordinary['events']),
        'author_event_count': len(author['events']),
        'copied_event_count': min(8, len(author['events'])),
        'scope': 'Copy the author notebook and visible recent events. The complete source '
                 'history remains in its existing proposal session; the previous ordinary '
                 'history is retained, and all past costs remain charged.'})
    ordinary['notes'] = author['notes']
    ordinary['events'].extend(deepcopy(author['events'][-8:]))
    assert ordinary['notes'] == author['notes']
    assert ordinary['events'][-8:] == author['events'][-8:]
    params = deepcopy(params)
    params['operator'] = 'inquire'
    return params


def run(branch='live'):
    OUT = BASE / branch
    source = (BASE / 'ordinary_author_20261010/result.json' if branch == 'ordinary_parent_resume_20261010'
              else BASE / 'decision_resume_20261010/result.json' if branch in (
                  'recursive_author_20261010', 'ordinary_author_20261010')
              else BASE / 'synchronized_rules_20261010/result.json' if branch == 'decision_resume_20261010'
              else BASE / 'authored_rules_20261010/result.json' if branch == 'synchronized_rules_20261010'
              else BASE / 'author_return_20261010/result.json' if branch == 'authored_rules_20261010'
              else BASE / 'author_20261010/result.json' if branch == 'author_return_20261010'
              else BASE / 'extension/result.json' if branch == 'author_20261010'
              else BASE / 'sustained/result.json' if branch == 'extension'
              else BASE / 'parent_interface/result.json' if branch == 'sustained'
              else BASE / 'returned/result.json' if branch == 'parent_interface'
              else BASE / 'retry/result.json' if branch == 'returned'
              else BASE / 'continued/result.json' if branch == 'retry'
              else BASE / 'live/result.json' if branch == 'continued' else SOURCE)
    agent, lab, context = restore_saved(source, OUT,
        additional_calls=(16 if branch in ('authored_rules_20261010', 'synchronized_rules_20261010') else
                          8 if branch in ('live', 'sustained', 'extension') else 0),
        python_seconds=1200 if branch == 'authored_rules_20261010' else None,
        additional_wall_seconds=1800 if branch in (
            'authored_rules_20261010', 'synchronized_rules_20261010') else 0)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if branch in ('author_20261010', 'author_return_20261010', 'authored_rules_20261010',
                  'synchronized_rules_20261010', 'decision_resume_20261010',
                  'recursive_author_20261010', 'ordinary_author_20261010', 'ordinary_parent_resume_20261010'):
        protocol.pop('retry_reason', None)
    if branch == 'authored_rules_20261010':
        prior_steps = agent.limits['environment_steps']
        agent.limits['environment_steps'] = prior_steps + 128
        protocol['environment_budget_extension'] = {'from_steps': prior_steps,
            'to_steps': agent.limits['environment_steps'],
            'origin': 'operator_requested_continuation'}
        protocol['limits'] = deepcopy(agent.limits)
    if branch == 'author_return_20261010':
        protocol['retry_reason'] = (
            'The preceding provider request exceeded its input character limit, without a timeout. '
            'Resume the interrupted author using semantic previews and exact saved-data references, '
            'retaining the failed attempt, all paid observations, and the original remaining allowance.')
    if branch == 'retry':
        agent.limits['provider_request_timeout_seconds'] = 300
        protocol['limits'] = deepcopy(agent.limits)
        protocol['retry_reason'] = ('The preceding request timed out before producing a decision. '
            'Retain its charged attempt and all remaining task budgets; increase only the '
            'per-request timeout from 180 to 300 seconds. Current proposal assignment IDs '
            'now follow the caller without rewriting old answers or events.')
    protocol.update(study='condition_author_physical_practice',
        branch=branch,
        auth_mode=verify_codex_subscription(),
        model={'requested_alias': 'gpt-6-luna', 'reasoning_effort': 'medium'},
        scope='Resume the existing author and its 451-observation development history. '
              'The live branch adds eight calls to one unused call; the continued branch '
              'uses the same remaining allowance after repairing stale research advice. '
              'The retry increases only the per-request timeout after a provider timeout. '
              'The returned branch refreshes the active conditions on all paid observations '
              'and returns directly to the parent with its three unused calls; no child is reopened. '
              'The parent_interface branch uses the same two remaining calls after exposing '
              'the existing condition-research contract in the shared parent interface. '
              'The sustained branch adds eight calls after those calls are spent, retaining '
              'the same code, tools, candidates and all other ceilings. '
              'The extension branch adds eight calls to five unused calls and lets the normal '
              'task dispatcher open the existing researcher when finite target support is empty. '
              'The author_20261010 branch retains the eleven unused calls and resumes the existing '
              'computed-representation author through that dispatcher. Its original comparison is '
              'recomputed on all paid responses before its first decision; edits refresh associated '
              'finite conditions before the next decision. No additional allowance is added. '
              'The author_return_20261010 branch resumes the interrupted author from its saved '
              'host action after fixing duplicate request content and post-trial source comparison '
              'return. It retains the six unused calls, all drafts and all other ceilings. '
              'The authored_rules_20261010 branch reopens the same source author with the '
              'existing extension assignment after connecting explicit Boolean candidate_additions '
              'to the retained condition bank. It adds 16 calls, 128 world steps, 300 cumulative '
              'Python seconds and 1800 wall seconds. The author receives 12 of those calls before '
              'returning to the parent. No proposed law is supplied by this branch. '
              'The synchronized_rules_20261010 branch continues its 538-observation endpoint '
              'after repairing source/bank view synchronization and binding-derived trial return. '
              'It adds 16 model calls and 1800 wall seconds, retaining the world and Python ceilings. '
              'The existing condition bank is refreshed on paid evidence before the parent resumes; '
              'no author is manually reopened and no scientific assignment is added. '
              'The decision_resume_20261010 branch retains the remaining allowance following '
              'the synchronized branch input-too-large failure, reusing semantic request previews '
              'and resuming budget-interrupted condition decisions on their already screened rules. '
              'The recursive_author_20261010 branch reopens the same source author at its '
              '563-observation endpoint with its existing assignment, exposing explicit child '
              'inquire and return to the same draft. It retains all remaining allowances, '
              'reserves the author decision after a child and then returns to the parent. '
              'The author can choose its condition experiment through the retained controller, '
              'receive actual responses, and continue or return to the parent. '
              'Finite target-consequence projections use paid measurements and frozen rules. '
              'No scientific draft, physical point, interpretation or hidden answer is supplied '
              'by the runner. This operator-requested continuation is not a matched benchmark.')
    write_json(OUT / 'protocol.json', protocol)
    if branch == 'ordinary_author_20261010':
        protocol.update(study='ordinary_loop_at_the_same_author_checkpoint',
            paired_branch=str(BASE / 'recursive_author_20261010'),
            scope='Restore the same 563-observation decision_resume checkpoint as the '
                  'recursive_author branch, with identical model, assignment, source files '
                  'and ceilings (ten remaining calls, eight for the selected local loop). '
                  'Refresh the same source comparison and condition bank, and hand the '
                  'ordinary loop the author notebook and last eight visible events while '
                  'retaining both histories. Select existing inquire instead of '
                  'propose_explanations; the ordinary tool policy and subsequent feedback '
                  'are the treatment. The first local return is the primary diagnostic '
                  'endpoint; both branches subsequently resume the normal parent loop. '
                  'One development case, not a general performance or recursion comparison. '
                  'No scientific hypothesis, setting or new measurement is supplied.')
        write_json(OUT / 'protocol.json', protocol)
    if branch == 'ordinary_parent_resume_20261010':
        protocol.update(study='ordinary_loop_parent_return_after_context_repair',
            paired_branch=str(BASE / 'recursive_author_20261010'),
            scope='Continue the ordinary branch after its eighth local request failed at '
                  'the provider character limit. The attempted request remains charged '
                  'and the eight-call local allowance is spent; do not reopen that local '
                  'loop. Repair only the existing preview of older nested proposal returns '
                  'and give the two remaining episode calls to the normal parent dispatcher, '
                  'as reserved in both arms. Retain both new physical trials, all drafts and '
                  'the original task ceilings. The local endpoint remains provider_failed; '
                  'this continuation does not turn that failure into a voluntary stop.')
        write_json(OUT / 'protocol.json', protocol)
    agent.provider = CodexCodeProvider(agent.workspace, model='gpt-6-luna',
        reasoning_effort='medium', timeout_seconds=agent.limits['provider_request_timeout_seconds'])

    def checkpoint(data):
        event = data['events'][-1] if len(data['events']) > context['prefix_events'] else {}
        brief = {'status': 'running',
            'new_model_calls': data['costs']['model_calls'] - context['costs_before']['model_calls'],
            'new_observations': len(agent.observations) - context['prefix_observations'],
            'latest_action': event.get('action'), 'latest_operator': event.get('arguments', {}).get('operator')}
        write_json(OUT / 'active.json', brief)
        print(json.dumps(brief), flush=True)

    agent._on_checkpoint = checkpoint
    if branch in ('returned', 'parent_interface', 'sustained', 'extension', 'author_20261010',
                  'author_return_20261010', 'authored_rules_20261010', 'synchronized_rules_20261010',
                  'decision_resume_20261010', 'recursive_author_20261010', 'ordinary_author_20261010',
                  'ordinary_parent_resume_20261010'):
        path = agent.research_advice['proposal_path']
        conditions = refresh_condition_proposal(agent, path)
        write_json(OUT / 'paid_return.json', conditions)
        agent.events.append({'action': 'condition_return', 'arguments': {'path': path},
            'trigger_origin': 'operator_refreshes_paid_evidence_before_parent_continuation',
            'feedback': {'condition_research': deepcopy(conditions)}})
        if branch in ('synchronized_rules_20261010', 'decision_resume_20261010',
                      'recursive_author_20261010', 'ordinary_author_20261010',
                      'ordinary_parent_resume_20261010'):
            packet = response_premise_proposal_inputs(agent, path)
            source_view = agent.llm_inquiries['proposal:' + packet['binding_path']]['proposal']
            if source_view.get('condition_research') != conditions:
                raise ValueError('Source and condition-bank views differ before live continuation')
            write_json(OUT / 'starting_views.json', {
                'source_path': packet['binding_path'], 'condition_path': path,
                'views_equal': True, 'response_count': len(packet['responses']),
                'selected_experiment': conditions.get('selected_experiment'),
                'authored_rules': [{key: rule[key] for key in
                    ('id', 'authored_id', 'status', 'conflict_count', 'conflicting_responses')
                    if key in rule} for rule in conditions['candidates']
                    if rule.get('origin') == 'authored']})
        if branch in ('author_return_20261010', 'authored_rules_20261010',
                      'recursive_author_20261010', 'ordinary_author_20261010'):
            previous = json.loads(source.read_text())
            interrupted = next(event for event in reversed(previous['events'])
                if event.get('trigger_origin') == 'host_condition_language_extension')
            params = deepcopy(interrupted['arguments'])
            if branch == 'authored_rules_20261010':
                # The old empty-support trigger is historical: later paid
                # returns can make the current projection unresolved instead.
                params['question']['question'] = (
                    'Develop executable competing response hypotheses for the unfinished task '
                    'using the retained measurements, trials and current condition_research. '
                    'Earlier empty-support diagnoses are historical; use the returned current '
                    'comparisons and projections to decide what needs revision. Compute proposed '
                    'rules and their applicability, compare them with the paid counterexamples, '
                    'and choose a distinguishing experiment when available. The current response '
                    'binding interface accepts explicit Boolean candidate_additions alongside '
                    'its estimator comparison. Retain the old rules and their evidence. Choose '
                    'the scientific content yourself and return the computed results or the '
                    'specific unresolved prerequisite to the parent.')
            params['max_model_calls'] = min(12 if branch == 'authored_rules_20261010' else params['max_model_calls'],
                agent.limits['max_model_calls'] - agent.costs['model_calls'] - 1)
            if branch == 'ordinary_author_20261010':
                params = ordinary_author_handoff(agent, lab, params, OUT)
                write_json(OUT / 'local_assignment.json', params)
            outcome = agent._execute('run', params, lab)
            if branch == 'ordinary_author_20261010':
                write_json(OUT / 'local_return.json', {key: deepcopy(outcome[key]) for key in
                    ('operator', 'status', 'question', 'model_calls', 'notes')})
            agent.events.append({'action': 'run', 'arguments': params,
                'trigger_origin': ('operator_runs_ordinary_loop_with_author_memory'
                    if branch == 'ordinary_author_20261010' else 'operator_reopens_author_with_recursive_inquiry'
                    if branch == 'recursive_author_20261010' else
                    'operator_reopens_author_with_explicit_condition_interface'
                    if branch == 'authored_rules_20261010' else 'operator_resumes_interrupted_author'),
                'feedback': deepcopy(outcome)})
    else:
        comparison = agent.llm_inquiries['proposal:shared/response_representation_proposal.py']['proposal']['comparison_result']
        action = response_premise_design_action(comparison)
        action['arguments']['max_model_calls'] = {'live': 6, 'continued': 5, 'retry': 4}[branch]
        proposed = agent._execute(action['action'], action['arguments'], lab)
        write_json(OUT / 'proposal.json', proposed)
        agent.events.append({'action': action['action'], 'arguments': action['arguments'],
            'trigger_origin': 'operator_reopens_existing_premise_author', 'feedback': deepcopy(proposed)})
    agent._save_state()
    result = agent.run(lab, on_checkpoint=checkpoint)
    agent._save_state()
    result = account(agent)
    result.update(study_status='finished', episode_status=result['status'],
        execution={'operator': 'task_continuation', 'status': result['status']},
        costs_before=context['costs_before'],
        new_costs={key: value - context['costs_before'][key] for key, value in result['costs'].items()},
        observations=deepcopy(agent.observations), replay=context['replay'])
    write_json(OUT / 'result.json', result)
    brief = {'status': 'finished', 'episode_status': result['status'],
             'score': result['score'], 'new_costs': result['new_costs']}
    write_json(OUT / 'active.json', brief)
    print(json.dumps(brief), flush=True)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'live')
