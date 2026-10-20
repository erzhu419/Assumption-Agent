#!/usr/bin/env python3
"""Development study of authored representations and prospective experiments.

All arms receive identical paid prefixes and the same optional executable archive.
The provider sees the public trajectory interface, not the simulator family or a
menu of fitted state models. Old paper cohorts and their memory arms are unchanged.

Registered arms archive/fixed_pr2/pr3 additionally require
research_organization={phase:'source'|'target', theories:[shared old declarations],
initial_library_path:optional frozen file/reference, consolidation:{the existing
source_theory_ids/episode_sources/materials/allocation/mission arguments}}.
Source mission declares its stable mission_id and theory compute suballocation;
target instead requires pr3_library_path. These libraries govern automatic
organization, while all ordinary coding and explicit research tools remain shared.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.code_agent import CodexCodeProvider, evaluate_python_model
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.proposers import _events
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.research_episodes import extract_episodes, load_episode_sources, _contains_reference, _path
from hegel_machine.cognition.theory_consolidation import (
    LIBRARY_PATH, install_theory_library, _save as save_theory_library,
)
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json


ORGANIZATION_ARMS = ('archive', 'fixed_pr2', 'pr3')
ORGANIZATION_ORIGIN = 'registered_research_organization'


class RegisteredOrganizationAgent(CommonArchiveInquiryAgent):
    """Use the same strong Common condition; vary only automatic organization.

    B/C route the actual active task through PR2. C may additionally consolidate
    the preregistered source inputs under their retained theory suballocation.
    A keeps all ordinary tools and the original experiment-selection policy.
    """

    def __init__(self, *args, arm, research_organization, **kwargs):
        self.organization_group = arm
        self._organization_registration = deepcopy(research_organization)
        super().__init__(*args, arm='code_react', **kwargs)
        root = self._open_question({'id': 'task', 'question':
            'Infer the process from current paid observations and predict new public trajectories.',
            'parent_id': None, 'depends_on': []})
        root['origin'] = 'public_task'
        self._active_question_id = root['id']
        register_theories(self, research_organization.get('theories', []))
        phase = research_organization['phase']
        selected = research_organization['initial_library_path']
        if arm == 'pr3' and phase == 'target':
            selected = research_organization['pr3_library_path']
            self.theory_consolidation = install_theory_library(
                self, {'file': selected, 'json_path': []})
            save_theory_library(self)
        elif arm == 'pr3':
            selected = LIBRARY_PATH
        self._organization_state = {'group': arm, 'inner_arm': self.arm, 'phase': phase,
            'initial_library_path': research_organization['initial_library_path'],
            'automatic_library_path': selected,
            'theory_mission': deepcopy(research_organization.get('consolidation', {}).get('mission', {})),
            'automatic_event_indices': [], 'automatic_compute_cost': 0}
        selected_ids = research_organization.get('consolidation', {}).get('source_theory_ids', list(self.theory_index))
        self._organization_state['source_theory_scopes'] = {
            identifier: {key: deepcopy(self.theory_index[identifier][key]) for key in
                ('source_ref', 'episode_ref', 'program_source_ref') if key in self.theory_index[identifier]}
            for identifier in selected_ids if identifier in self.theory_index}

    def restore_research(self, previous, *args, **kwargs):
        # A provider-free run has a real checkpoint but no paid model request.
        source = Path(args[0])
        restore = (super().restore_practice_research if previous['costs']['model_calls'] == 0
                   and not (source / 'research_context.json').is_file() else super().restore_research)
        restored = restore(previous, *args, **kwargs)
        current = self._organization_state
        saved = previous['research_organization']
        for field in ('group', 'phase', 'inner_arm'):
            if saved[field] != current[field]:
                raise ValueError(f'cohort recovery must preserve organization {field}')
        self._organization_state = deepcopy(saved)
        self._organization_state.setdefault('source_theory_scopes', current['source_theory_scopes'])
        if self.organization_group == 'pr3' and saved['phase'] == 'source':
            self._organization_state['automatic_library_path'] = LIBRARY_PATH
        return restored

    def _automatic_work(self, arguments, laboratory, kind):
        before = self.costs['compute_cost']
        try:
            feedback = self._execute('run', arguments, laboratory)
        except (ValueError, TypeError, KeyError, OSError) as error:
            feedback = {'operator': arguments['operator'], 'status': 'automatic_research_failed',
                        'error': f'{type(error).__name__}: {error}',
                        'new_compute_cost': self.costs['compute_cost'] - before}
        actual = self.costs['compute_cost'] - before
        state = self._organization_state
        if feedback.get('cache_hit') and not actual and kind in state:
            return False
        event_index = len(self.events)
        self.events.append({'step': self.costs['model_calls'], 'action': 'run',
            'arguments': deepcopy(arguments), 'feedback': deepcopy(feedback),
            'origin': ORGANIZATION_ORIGIN,
            'usage': {'backend': ORGANIZATION_ORIGIN, 'total_tokens': 0}})
        state['automatic_event_indices'].append(event_index)
        state['automatic_compute_cost'] += actual
        state[kind] = {'status': feedback.get('status'), 'event_index': event_index,
                       'new_compute_cost': actual}
        return True

    def _registered_source_ids(self):
        """Extend explicit source Episodes, retaining exact IDs without provenance."""
        scopes = self._organization_state['source_theory_scopes']
        references = self._organization_registration.get('consolidation', {}).get('episode_sources', [])
        retained = load_episode_sources(references, self.workspace) if references else []
        episodes = {identifier: episode for episode in extract_episodes(
            self, retained_episodes=retained)['episodes'] for identifier in episode['theory_ids']}

        def contract(theory):
            reference = theory.get('source_ref') or theory.get('episode_ref', {})
            owner = reference.get('question_id')
            if owner is None:
                return None
            return owner, reference.get('meaning_version', 1)

        def address(theory):
            return theory.get('program_source_ref') or theory.get('episode_ref', {}).get('source_ref')

        def admitted_address(seed, theory):
            original = seed.get('registered_episode_source_ref') or address(seed)
            current = address(theory)
            if original is None or current is None:
                return original is current
            parent = next((ref for ref in references if _contains_reference(ref, original)), original)
            if not _contains_reference(parent, current):
                return False
            cutoff = parent.get('action_cutoff')
            if cutoff is not None:
                program_ref = theory.get('program_source_ref')
                if not program_ref or not _contains_reference(parent, program_ref):
                    return False
                tail = _path(program_ref)[len(_path(parent)):]
                return len(tail) >= 2 and tail[0] == 'actions' and type(tail[1]) is int and tail[1] <= cutoff
            return True

        for identifier, seed in scopes.items():
            episode = episodes.get(identifier, {})
            identity = contract(seed)
            if (identity is not None and identity == contract(self.theory_index.get(identifier, {}))
                    and identity[1] == episode.get('meaning_version')
                    and (not address(seed) or episode.get('source_ref') and
                         _contains_reference(episode['source_ref'], address(seed)))):
                seed.setdefault('episode_id', episode['episode_id'])
                seed.setdefault('source_id', episode['source_id'])
                if episode.get('source_ref'):
                    seed.setdefault('registered_episode_source_ref', deepcopy(episode['source_ref']))
        selected = []
        for identifier, theory in self.theory_index.items():
            if theory.get('consolidation_ref') and identifier not in scopes:
                continue
            if identifier in scopes and contract(scopes[identifier]) is None:
                selected.append(identifier)
                continue
            episode = episodes.get(identifier)
            identity = contract(theory)
            if (identifier in scopes and episode is not None and contract(scopes[identifier]) == identity
                    and identity[1] == episode['meaning_version'] and address(theory) == address(scopes[identifier])):
                selected.append(identifier)
                continue
            if episode is not None and identity is not None and any(contract(seed) == identity
                    and identity[1] == episode['meaning_version'] and admitted_address(seed, theory)
                    and seed.get('episode_id') == episode['episode_id']
                    and seed.get('source_id') == episode['source_id']
                    for seed in scopes.values() if contract(seed) is not None):
                selected.append(identifier)
        return selected

    def _consolidate_registered_source(self, laboratory):
        opened = False
        state, registration = self._organization_state, self._organization_registration
        source_work = registration.get('consolidation')
        if self.organization_group == 'pr3' and state['phase'] == 'source' and source_work:
            selected = self._registered_source_ids()
            # Host organization events are not a new source research attempt.
            basis = {'source_versions': [{key: deepcopy(self.theory_index.get(identifier, {}).get(key))
                       for key in ('id', 'version', 'source_ref')}
                       for identifier in selected],
                'research_events': sum(event.get('origin') != ORGANIZATION_ORIGIN for event in self.events),
                'paid_records': len(self.observations), 'revisions': len(self.revisions)}
            if state.get('consolidation_basis') != basis:
                active_id = self._active_question_id
                owner_id = 'organization:source:' + state['theory_mission']['mission_id']
                owner = next((q for q in self.questions if q['id'] == owner_id), None)
                if owner is None:
                    owner = self._open_question({'id': owner_id,
                        'question': 'Construct shared structure from the registered source episodes; '
                            'retain failures, costs and unresolved author work.',
                        'parent_id': None, 'depends_on': [], 'queue': 'theory',
                        'mission': deepcopy(state['theory_mission'])})
                arguments = {**deepcopy(source_work), 'operator': 'theory_consolidation',
                             'source_theory_ids': selected,
                             'research_question': {'id': owner_id},
                             'mission': deepcopy(state['theory_mission'])}
                opened = self._automatic_work(arguments, laboratory, 'consolidation') or opened
                state['theory_mission'] = deepcopy(owner['mission'])
                state['theory_question_id'] = owner_id
                self._active_question_id = active_id
                state['consolidation_basis'] = basis
        return opened

    def _before_model_turn(self, laboratory):
        opened = super()._before_model_turn(laboratory)
        if self.organization_group == 'archive' or self.model is not None:
            return opened
        opened = self._consolidate_registered_source(laboratory) or opened
        state = self._organization_state
        active = next((question for question in self.questions
                       if question['id'] == self._active_question_id), None)
        origin = (active or {}).get('origin')
        if (active is None or active.get('queue') == 'theory'
                or isinstance(origin, dict) and origin.get('kind') == 'multi_episode_commonality'):
            active = next(question for question in self.questions if question['id'] == 'task')
        arguments = {'operator': 'research_route', 'research_question': {'id': active['id']},
            'library_path': state['automatic_library_path'],
            'mission': {'purpose': 'task', 'mission_id': 'registered_task_organization'}}
        return self._automatic_work(arguments, laboratory, 'routing') or opened

    def _after_model_turn(self, laboratory):
        super()._after_model_turn(laboratory)
        self._consolidate_registered_source(laboratory)

    def _program_state(self):
        state = super()._program_state()
        if hasattr(self, '_organization_state'):
            state['research_organization'] = deepcopy(self._organization_state)
        return state

    def _request(self, laboratory):
        request = super()._request(laboratory)
        request['research_organization'] = deepcopy(self._organization_state)
        return request

    def result(self, status):
        result = super().result(status)
        result['research_organization'] = deepcopy(self._organization_state)
        return result


def _prepare_organization(plan, output, knowledge, previous_study=None):
    """Anchor the small registered libraries once, keeping all old knowledge."""
    arms = set(plan['arms'])
    if not arms.intersection(ORGANIZATION_ARMS):
        return None, knowledge
    if arms != set(ORGANIZATION_ARMS):
        raise ValueError('register archive, fixed_pr2 and pr3 together')
    registration = deepcopy(plan.get('research_organization') or {})
    if registration.get('phase') not in {'source', 'target'}:
        raise ValueError('three-arm organization needs a registered source or target phase')
    for case in plan['cases']:
        if 'arms' in case and set(case['arms']) != arms:
            raise ValueError('each source must retain all three registered organization groups')
    source_work = registration.get('consolidation', {})
    if registration['phase'] == 'source':
        mission = source_work.get('mission', {})
        allowance = mission.get('theory_learning_allocation', mission.get('allocation', {}))
        allowance = allowance.get('compute') if isinstance(allowance, dict) else allowance
        if (mission.get('purpose') not in {'theory_learning', 'method_learning'}
                or not mission.get('mission_id') or type(allowance) is not int
                or not 0 <= allowance <= plan['limits']['compute_budget']):
            raise ValueError('source organization needs one registered bounded theory mission')
        if source_work.get('theories', registration.get('theories', [])) != registration.get('theories', []):
            raise ValueError('registered source declarations must be shared by all three groups')
    memory = deepcopy(knowledge or {})
    selected = registration.get('initial_library_path') or memory.get('theory_consolidation')
    if previous_study:
        original_input = json.loads((previous_study / 'initial_archive.json').read_text())
        selected = original_input['theory_consolidation']
    fixed = output / 'initial_theory_library.json'
    if selected:
        reference = ({'file': str(Path(selected).resolve()), 'json_path': []}
                     if isinstance(selected, str) else deepcopy(selected))
        write_json(fixed, {'source_ref': reference})
    else:
        document = memory.get('files', {}).get(LIBRARY_PATH)
        write_json(fixed, json.loads(document) if document else {'candidates': {}})
    # Continue the same registered frozen object; a new output shell is only a
    # reference, not a new selection-library identity.
    registration['initial_library_path'] = reference['file'] if previous_study else str(fixed)
    memory['theory_consolidation'] = (deepcopy(reference) if previous_study else
                                    {'file': str(fixed), 'json_path': []})
    if registration['phase'] == 'target':
        selected = registration.get('pr3_library_path')
        if not selected:
            raise ValueError('target organization needs the frozen PR3 library reference')
        reference = ({'file': str(Path(selected).resolve()), 'json_path': []}
                     if isinstance(selected, str) else deepcopy(selected))
        frozen = output / 'pr3_theory_library.json'
        write_json(frozen, {'source_ref': reference})
        registration['pr3_library_path'] = str(frozen)
    return registration, memory


def protocol(*, kinds=("memoryless", "delay", "hysteresis"), cases=1,
             arms=("code_react", "hegel"), seed=61011017, config=None,
             model="gpt-6-luna", timeout_seconds=300, archive=None,
             research_organization=None):
    plan = {
        "study": "anonymous_authored_revision_development",
        "model": model, "arms": list(arms),
        "cases": [{"id": f"case_{index:03d}", "kind": kind, "seed": seed + index * 1009,
                   "prefix_seed": seed + index * 1009 + 100003,
                   "confirmation_seed": seed + index * 1009 + 200003,
                   "evaluation_seed": seed + index * 1009 + 300007}
                  for index, kind in enumerate(kind for _ in range(cases) for kind in kinds)],
        "limits": asdict(config or InquiryLLMConfig(observation_budget=576,
            confirmation_rows=96, max_model_calls=48, token_budget=1200000,
            execution_seconds=300., compute_budget=250000)),
        "provider": {"backend": "codex_cli", "timeout_seconds": timeout_seconds,
                     "reasoning_effort": "medium"},
        "prefix_rows": 72, "archive": str(archive) if archive else None,
        "controls": "Both arms have the same code archive, paid prefix, Python and experiment tools. "
                    "Only Hegel requires an executable hypothesis/research-question manifest.",
        "discovery": "No supplied state-model menu, diagnostic label, fitted candidate, or research procedure. "
                     "Predictors, representations, experiment programs and readouts are provider-authored.",
        "evaluation": {"rows": 512, "trajectory_length": 48, "mse_threshold": 1e-10},
        "terminal_order": "All training stops before terminal labels are generated. "
                          "Submitted predictor files and their paid training view stay fixed.",
        "failure_policy": "Keep all initially registered sources and provider, research and confirmation failures.",
        "scope": "Deterministic development controls on 21 public input levels; learning horizon 24. "
                 "Terminal horizon 48 tests new paths, not different physical objects. "
                 "This pilot alone establishes neither recursive superiority nor cross-task meta-prior learning.",
    }
    if research_organization is not None:
        plan['research_organization'] = deepcopy(research_organization)
        plan['controls'] = ('All three groups use Common code_react, the same public task root, '
            'complete original archive, tools, budgets and terminal order. B/C automatically '
            'route actual task contracts through PR2. Only C consolidates registered source '
            'evidence. Fixed libraries constrain automatic organization; ordinary Python, '
            'explicit research tools and authored candidates remain available to every group. '
            'Target automatic libraries remain frozen.')
    return plan


def restore_pending_question_error(agent, laboratory, previous, workspace, *, model):
    """Return the saved, paid declaration error without rerunning scientific work."""
    workspace = Path(workspace)
    call = previous['costs']['model_calls'] + 1
    reply = workspace / f'code_action_{call:03d}.json'
    trace = workspace / f'code_call_{call:03d}.jsonl'
    payload = json.loads(reply.read_text())
    arguments = json.loads(payload['arguments_json'], strict=False)
    if payload.get('source_code') is not None:
        arguments['code'] = payload['source_code']
    if payload.get('research_question') is not None:
        arguments['research_question'] = payload['research_question']
    if 'research_notes' in payload:
        arguments['notes'] = payload['research_notes']
    declaration = arguments.get('research_question') or arguments.get('question')
    if (payload['action'] != 'evaluate' or arguments.get('operator') is not None
            or 'component' in arguments or arguments.get('target', 'response') != 'response'
            or not isinstance(declaration, str) or not declaration
            or previous.get('online_questions') or previous.get('model') is not None):
        raise ValueError('recover only the saved pre-execution question-declaration failure')
    events, _ = _events(trace.read_text())
    completion = next((event for event in reversed(events)
                       if event.get('type') == 'turn.completed'), None)
    if completion is None:
        raise ValueError('the pending paid reply has no completed usage record')
    usage = {'backend': 'codex_cli', 'model': model, 'calls': 1,
             **{name: completion['usage'].get(name)
                for name in ('input_tokens', 'output_tokens', 'cached_input_tokens')}}
    if not all(isinstance(usage[key], int) for key in ('input_tokens', 'output_tokens')):
        raise ValueError('the pending paid reply has no reported token usage')
    usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
    agent.costs['model_calls'] = call
    agent.costs['reported_tokens'] += usage['total_tokens']
    try:
        agent._execute(payload['action'], arguments, laboratory)
    except (ValueError, TypeError) as error:
        feedback = {'error': f'{type(error).__name__}: {error}'}
    else:
        raise RuntimeError('malformed question must return before scientific execution')
    agent.events.append({'step': call - 1, 'action': payload['action'],
                         'arguments': arguments, 'feedback': feedback, 'usage': usage})
    agent._after_model_turn(laboratory)
    agent.provider.calls = call
    return {'saved_reply': str(reply), 'saved_usage': str(trace), 'model_call': call,
            'reported_tokens': usage['total_tokens'],
            'scope': 'Saved paid reply charged once and returned as ordinary input feedback. '
                'No predictor write, evaluation, acquisition or extra allowance.'}


def run_study(plan, output, *, provider_factory, knowledge=None, prepare_prefix=None,
              laboratory_factory=None, workers=1, agent_factory=None,
              previous_study=None, resume_task=''):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a fresh output directory to retain earlier developmental failures")
    previous_study = Path(previous_study).resolve() if previous_study else None
    if previous_study:
        previous_plan = json.loads((previous_study / 'protocol.json').read_text())
        for field in ('cases', 'arms', 'model', 'provider', 'limits', 'prefix_rows', 'evaluation'):
            if plan[field] != previous_plan[field]:
                raise ValueError(f'cohort recovery must preserve registered {field}')
        if plan.get('research_organization') != previous_plan.get('research_organization'):
            raise ValueError('cohort recovery must preserve registered research_organization')
    organization, prepared_knowledge = _prepare_organization(plan, output, knowledge, previous_study)
    agent_factory = agent_factory or (RegisteredOrganizationAgent if organization else CommonArchiveInquiryAgent)
    write_json(output / "protocol.json", plan)
    # Literal common input is saved once; no arm learns the other's live results.
    write_json(output / "initial_archive.json", prepared_knowledge or {})
    limits = InquiryLLMConfig(**plan["limits"])
    if laboratory_factory is None:
        laboratory_factory = lambda case: AnonymousHistoryLab(kind=case["kind"], seed=case["seed"])
    jobs = []
    for case_index, case in enumerate(plan["cases"]):
        arms = plan["arms"]
        ordered = case.get("arms", arms[case_index % len(arms):] + arms[:case_index % len(arms)])
        jobs.extend((case, arm) for arm in ordered)

    def train(job):
        case, arm = job
        workspace = output / case["id"] / arm
        old_workspace = previous_study / case['id'] / arm if previous_study else None
        if old_workspace and (old_workspace / 'training.json').is_file():
            result = json.loads((old_workspace / 'training.json').read_text())
            return {'case': case, 'arm': arm, 'result': result, 'initial_event_count': 0}
        agent = agent_factory(provider_factory(workspace), workspace,
            arm=arm, knowledge=deepcopy(prepared_knowledge or {}), config=limits,
            confirmation_seed=case["confirmation_seed"], source_task=False if organization else not bool(knowledge),
            **({'research_organization': organization} if organization else {}))
        laboratory = laboratory_factory(case)
        prefix_account = {}
        if old_workspace and old_workspace.is_dir():
            previous = json.loads((old_workspace / 'active.json').read_text())
            agent.restore_research(previous, old_workspace, laboratory,
                                   confirmation_seed=case['confirmation_seed'])
            agent.common_interface += resume_task
            recovery = (None if organization else restore_pending_question_error(agent, laboratory, previous,
                old_workspace, model=plan['model']))
            prefix_account = {'prior_research_cost': deepcopy(agent.costs),
                              'resume_origin': str(old_workspace / 'active.json'),
                              **({'recovered_paid_reply': recovery} if recovery else {})}
            write_json(workspace / 'active.json', agent.result('researching'))
        elif prepare_prefix is None:
            prefix = laboratory.confirmation_actions(plan["prefix_rows"], Random(case["prefix_seed"]))
            feedback = agent._execute("query", {"actions": prefix}, laboratory)
            if feedback.get("error"):
                raise ValueError(feedback["error"])
            agent.events.append({"step": -1, "action": "query", "arguments": {"actions": prefix},
                "feedback": feedback, "usage": {"backend": "common_paid_prefix", "total_tokens": 0}})
        else:
            prefix_account = prepare_prefix(agent, laboratory, case)
        initial_event_count = len(agent.events)
        result = agent.run(laboratory, on_checkpoint=lambda data, path=workspace:
                           write_json(path / "active.json", data))
        if prefix_account:
            result.update(prefix_account)
            if "prior_research_cost" in prefix_account:
                prior = prefix_account["prior_research_cost"]
                result["continuation_cost"] = {key: value - prior.get(key, 0)
                    if value is not None else None for key, value in result["costs"].items()}
            else:
                historical = prefix_account["historical_prefix_cost"]
                result["additional_cost"] = {key: value - historical.get(key, 0)
                    if value is not None else None for key, value in result["costs"].items()}
        write_json(workspace / "training.json", result)
        write_json(workspace / "archive.json", result["outgoing_knowledge"])
        print(json.dumps({"case": case["id"], "arm": arm, "status": result["status"],
                          "costs": result["costs"]}), flush=True)

        return {"case": case, "arm": arm, "result": result,
                "initial_event_count": initial_event_count}

    if workers == 1:
        trials = [train(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # map retains registration order; all training finishes before any
            # terminal rows are generated by the separate evaluation loop.
            trials = list(executor.map(train, jobs))

    records = []
    for trial in trials:
        case, result = trial["case"], trial["result"]
        evaluation = {"status": "no_submitted_model", "mse": None}
        if result["model"]:
            rows = laboratory_factory(case).evaluation_rows(
                length=plan["evaluation"]["rows"], seed=case["evaluation_seed"])
            evaluation = evaluate_python_model(result["workspace"], result["model"], rows,
                                               timeout_seconds=limits.execution_seconds)
            evaluation.pop("predictions", None)
        evaluation["passed"] = (evaluation.get("returncode") == 0 and
            evaluation.get("mse", float("inf")) <= plan["evaluation"]["mse_threshold"])
        host_workspace = output / case['id'] / trial['arm']
        host_workspace.mkdir(parents=True, exist_ok=True)
        write_json(host_workspace / "evaluation.json", evaluation)
        usage = [event.get("usage", {}) for event in result["events"]
                 if event.get("usage", {}).get("backend") == "codex_cli"]
        records.append({"case": case["id"], "kind": case["kind"], "arm": trial["arm"],
            "status": result["status"], "model": result["model"], "costs": result["costs"],
            "fresh_confirmation_passed": (result.get("fresh_confirmation") or {}).get("passed", False),
            "terminal": evaluation,
            "revision_count": len(result.get("revision_proposals", [])),
            "cached_input_tokens": sum(item.get("cached_input_tokens") or 0 for item in usage),
            "calls_with_cache_hits": sum(bool(item.get("cached_input_tokens")) for item in usage),
            "archive": str(Path(result["workspace"]) / "archive.json")})
        if organization:
            records[-1]['research_organization'] = deepcopy(result['research_organization'])
        for field in ("historical_prefix_cost", "additional_cost", "prefix_origin",
                      "prior_research_cost", "continuation_cost", "resume_origin",
                      "recovered_paid_reply"):
            if field in result:
                records[-1][field] = deepcopy(result[field])
    live_events = [event for trial in trials
                   for event in trial["result"]["events"][trial["initial_event_count"]:]
                   if event.get("usage", {}).get("backend") not in
                       {"common_paid_prefix", "restored_paid_prefix", ORGANIZATION_ORIGIN}]
    summary = {"study": plan["study"], "status": "completed", "trials": records,
               "evidence_kind": "real_external_model" if live_events and all(
                   event.get("usage", {}).get("backend") == "codex_cli" for event in live_events)
                   else "scripted_interface_check",
               "source_count": len(plan["cases"]), "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", choices=("memoryless", "delay", "hysteresis"),
                        default=["memoryless", "delay", "hysteresis"])
    parser.add_argument("--cases", type=int, default=1, help="Independent source seeds per control family")
    parser.add_argument("--arms", nargs="+", choices=("code_react", "hegel"), default=["code_react", "hegel"])
    parser.add_argument("--archive", type=Path, help="Identical incoming executable archive for all arms")
    parser.add_argument("--seed", type=int, default=61011017)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--timeout-seconds", type=float, default=300.)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    plan = protocol(kinds=args.kinds, cases=args.cases, arms=args.arms, seed=args.seed,
                    model=args.model, timeout_seconds=args.timeout_seconds, archive=args.archive)
    plan["provider"]["auth_mode"] = verify_codex_subscription()
    knowledge = json.loads(args.archive.read_text()) if args.archive else None
    def provider(workspace):
        return CodexCodeProvider(workspace, model=args.model, reasoning_effort="medium",
                                 timeout_seconds=args.timeout_seconds)
    print(json.dumps(run_study(plan, args.output, provider_factory=provider, knowledge=knowledge), indent=2))


if __name__ == "__main__":
    main()
