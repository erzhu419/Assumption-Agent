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
The CLI requires explicit arms; --research-organization reads the registered
source/target JSON. Its optional exploration_policy is shared by B/C and retained
on recovery. Preparing a plan never authorizes acquisition on its own.
Its optional routing_allocation passes the existing construction allocation to
research_route, leaving the rest of the task budget available for preparation
and experiment selection.
An explicit author={kind:'finite_native',path:'shared/driver.py',max_steps:N,
confirm:bool} executes the same archived procedure in all three Common groups,
without constructing a provider. Source question scopes may be preregistered as
consolidation.source_question_refs=[{question_id,meaning_version}] before their
native candidate banks exist. Scientific observations still use the task budget.
New multi-source plans reserve the final cohort consolidation inside C's original
combined compute cap via source_cohort_allocation; local hooks retain their source
basis with zero consolidation compute. Plans without that contract keep their
registered allocation on recovery.
run_target_intervention reuses a completed target ABC study's common initial
input and runs only fresh C jobs for a frozen removal manifest. The full ABC
plan, old library, current runtime, budgets and confirmation contract stay
matched; prior target results are referenced only in the final paired readout.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import json
from math import isfinite
import os
from pathlib import Path
from random import Random
from shutil import copy2
import sys
from time import monotonic
from traceback import format_exc
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.anonymous_method_learning import compact_workspace_result
from hegel_machine.cognition.anonymous_research_archive import resolve_workspace_archive
from hegel_machine.cognition.code_agent import (
    CodexCodeProvider, evaluate_python_model, archived_paid_trace_references,
)
from hegel_machine.cognition.checkpoint_storage import clone_or_copy
from hegel_machine.cognition.experiment_runtime import (
    physical_worker_cpus as _native_worker_cpus,
    physical_cpu_groups as _native_worker_cpu_groups,
    run_process_trials as _native_process_trials,
)
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.proposers import _events
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.question_selection import validate_exploration_policy
from hegel_machine.cognition.research_episodes import (
    extract_episodes, load_episode_sources, retained_revision_sources,
    _contains_reference, _path, _question_versions,
)
from hegel_machine.cognition.theory_consolidation import (
    LIBRARY_PATH, install_theory_library, save_theory_index,
    _save as save_theory_library, _episode_file_versions,
)
from hegel_machine.cognition.temporal_paired_method import execute_procedure
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
        incoming = kwargs.get('knowledge')
        if isinstance(incoming, dict) and 'workspace' in incoming and 'files' not in incoming:
            kwargs['knowledge'] = resolve_workspace_archive(incoming)
            if kwargs.get('paid_trace_references') is None:
                workspace = Path(kwargs['workspace'] if 'workspace' in kwargs else args[1])
                kwargs['paid_trace_references'] = archived_paid_trace_references(
                    kwargs['knowledge']['files'], workspace / 'reference_archive',
                    source_workspace=incoming['workspace'])
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
            'exploration_policy': deepcopy(research_organization.get('exploration_policy')),
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
        if saved.get('exploration_policy') != current.get('exploration_policy'):
            raise ValueError('cohort recovery must preserve exploration_policy')
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
                        'traceback': format_exc(limit=-8),
                        'new_compute_cost': self.costs['compute_cost'] - before}
            if isinstance(error, OSError):
                feedback['errno'] = error.errno
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
        """Extend registered source Episodes using their actual retained provenance."""
        scopes = self._organization_state['source_theory_scopes']
        references = self._organization_registration.get('consolidation', {}).get('episode_sources', [])
        retained = load_episode_sources(references, self.workspace) if references else []
        intake = extract_episodes(self, retained_episodes=retained)
        episodes = {identifier: episode for episode in intake['episodes']
                    for identifier in episode['theory_ids']}
        revision_sources = retained_revision_sources(self, intake['episodes'])

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

        question_scopes = self._organization_registration.get('consolidation', {}).get('source_question_refs', [])
        question_versions = _question_versions(self)

        def registered_native_source(theory, episode):
            source, reference = theory.get('source_ref', {}), theory.get('program_source_ref', {})
            if source.get('kind') != 'native_candidate' or not reference.get('file') or reference.get('archive'):
                return False
            identity = source.get('question_id'), source.get('meaning_version', 1)
            if not any(identity == (ref['question_id'], ref['meaning_version']) for ref in question_scopes):
                return False
            if any(type(source.get(key)) is not int or source[key] < 0
                   for key in ('round_index', 'candidate_index')):
                return False
            file = Path(reference['file'])
            if (file if file.is_absolute() else self.workspace / file).resolve() != (self.workspace / 'research_state.json').resolve():
                return False
            retained = question_versions.get(identity)
            if retained is None:
                return False
            question, question_ref = retained
            tail = ['native_condition_research', source.get('round_index'),
                    'candidates', source.get('candidate_index'), 'program']
            owner_path = _path(question_ref)
            # Old index records keep their original address; the source meaning
            # selects its retained bank after a question changes meaning.
            if _path(reference) not in ([*owner_path, *tail], [*owner_path[:2], *tail]):
                return False
            try:
                candidate = question['native_condition_research'][tail[1]]['candidates'][tail[3]]
            except (IndexError, KeyError, TypeError):
                return False
            return (candidate.get('program') is not None and episode['meaning_version'] == identity[1]
                and episode['question_ref'] == question_ref)

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
            episode = episodes.get(identifier)
            revision = revision_sources.get(identifier)
            if (revision is not None and episode is not None
                    and episode['question_ref'] == revision['question_ref']
                    and episode['source_id'] == revision['source_id']
                    and any((reference['question_id'], reference['meaning_version']) ==
                            (revision['episode_ref']['question_id'],
                             revision['episode_ref']['meaning_version'])
                            for reference in question_scopes)):
                # Actual local adoption can supply a typed after; its inherited
                # library provenance does not create another physical source.
                selected.append(identifier)
                continue
            if theory.get('consolidation_ref') and identifier not in scopes:
                continue
            if identifier in scopes and contract(scopes[identifier]) is None:
                selected.append(identifier)
                continue
            episode = episodes.get(identifier)
            identity = contract(theory)
            if episode is not None and registered_native_source(theory, episode):
                selected.append(identifier)
                continue
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
        if not (isinstance(origin, dict) and origin.get('kind') == 'unfinished_parent_revision'):
            self._bind_revision_target(active)
        arguments = {'operator': 'research_route', 'research_question': {'id': active['id']},
            'library_path': state['automatic_library_path'],
            'mission': {'purpose': 'task', 'mission_id': 'registered_task_organization'}}
        if state.get('exploration_policy') is not None:
            arguments['exploration_policy'] = deepcopy(state['exploration_policy'])
        allocation = self._organization_registration.get('routing_allocation')
        if allocation is not None:
            arguments['allocation'] = deepcopy(allocation)
        from hegel_machine.cognition.finite_native_author import (
            export_revision_candidates, ordinary_completion_reserve,
        )

        reserve = ordinary_completion_reserve(self, active)
        original_config = self.config
        if reserve is not None:
            start = self.costs['compute_cost']
            self.config = replace(original_config, compute_budget=max(
                start, original_config.compute_budget - reserve['compute_cost']))
            available = max(0, self.config.compute_budget - start)
            arguments['allocation'] = {
                **arguments.get('allocation', {}),
                'compute': min(arguments.get('allocation', {}).get('compute', available), available),
                'nominal_compute_budget': original_config.compute_budget,
                'comparison_rows': reserve['paid_rows']}
        try:
            opened = self._automatic_work(arguments, laboratory, 'routing') or opened
        finally:
            if reserve is not None:
                reserve['new_candidate_comparison_compute'] = (
                    max(start, original_config.compute_budget - reserve['compute_cost'])
                    - self.config.compute_budget)
                reserve['route_compute_cost'] = self.costs['compute_cost'] - start
                self.config = original_config
                active['ordinary_completion_reserve'] = reserve
                self._prepare_program_state()

        routing = state.get('routing')
        track = (self.events[routing['event_index']]['feedback'].get('research_track_question_ref')
                 if routing is not None else None)
        export_owner = (_question_versions(self)[track['question_id'], track['meaning_version']][0]
                        if track is not None else active)
        exported = export_revision_candidates(self, export_owner)
        if exported['changed']:
            self._prepare_program_state()
        return opened or exported['new_exports'] > 0

    def _research_candidate_parent(self, question, target):
        """Take one measured before in original round/registration order.

        This research choice neither adopts a predictor nor examines the learned
        edit's applicability. Only the complete current paid fitting scope is
        eligible; a newer observation leaves the old failure at its old scope.
        """
        n, records = len(self.rows), len(self.observations)
        if not n:
            return None
        row_indices = list(range(n))
        question_index = self.questions.index(question)
        for round_index, round_ in enumerate(question.get('native_condition_research', [])):
            if (round_.get('observed_record_count') != records
                    or round_.get('fit_record_indices') != list(range(records))
                    or round_.get('fit_evidence') != row_indices):
                continue
            for registered in round_.get('native_theory_registration', {}).get('registered', []):
                theory = self.theory_index.get(registered['theory_id'])
                if theory is None or registered.get('version') != theory.get('version'):
                    continue
                source = theory.get('source_ref', {})
                if (source != registered.get('source_ref')
                        or source.get('kind') != 'native_candidate'
                        or source.get('question_id') != question['id']
                        or source.get('meaning_version', 1) != question.get('meaning_version', 1)
                        or source.get('round_index') != round_index
                        or any(theory.get(key) != target.get(key) for key in
                               ('output_unit', 'context_ref'))
                        or any(target.get('role_signature', {}).get(name) != contract
                               for name, contract in theory.get('role_signature', {}).items())):
                    continue
                index = source['candidate_index']
                candidate = round_['candidates'][index]
                base = ['questions', question_index, 'native_condition_research', round_index]
                candidate_path = [*base, 'candidates', index]
                program_ref = {'file': 'research_state.json',
                               'json_path': [*candidate_path, 'program']}
                mse = candidate.get('fit_mse')
                if (not candidate.get('program') or theory.get('program_source_ref') != program_ref
                        or registered.get('program_source_ref') != program_ref
                        or candidate.get('fit_evidence') != row_indices
                        or not isinstance(mse, (int, float)) or not isfinite(mse)
                        or mse <= self.config.confirmation_mse_threshold):
                    continue
                return {'theory_id': theory['id'], 'version': theory['version'],
                    'selection_rule': 'registered_round_order_first_complete_paid_failure',
                    'source_ref': deepcopy(source), 'program_source_ref': program_ref,
                    'fit_mse_ref': {'file': 'research_state.json',
                                    'json_path': [*candidate_path, 'fit_mse']},
                    'fit_evidence_ref': {'file': 'research_state.json',
                                         'json_path': [*candidate_path, 'fit_evidence']},
                    'round_scope_ref': {'file': 'research_state.json', 'json_path': base},
                    'rows': n, 'records': records, 'fit_mse': mse,
                    'scope': 'current_complete_paid_prefix_not_confirmation',
                    'incumbent_source_model': (self.evaluated_model or {}).get('source_model'),
                    'incumbent_ref': {'file': 'research_state.json', 'json_path': ['evaluated_model']}}
        return None

    def _retained_revision_parent(self, question, target):
        """Resolve this meaning's actual retained fixed export to its exact Program."""
        from hegel_machine.cognition.language import Program
        from hegel_machine.cognition.research_episodes import resolve_program_sources
        from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
        from hegel_machine.cognition.theory_consolidation import _absolute_reference

        retained = self.evaluated_model or {}
        selected = retained.get('source_model')
        matches = [item for item in question.get('revision_candidate_exports', [])
                   if item.get('model') == selected]
        if len(matches) != 1 or not retained.get('model'):
            return None
        exported = matches[0]
        reference = exported.get('theory_ref', {})
        theory = self.theory_index.get(reference.get('id'))
        if theory is None or theory.get('version') != reference.get('version'):
            return None
        program_ref = exported.get('program_source_ref', {})
        expected_path = ['theory_index', reference['id'], 'program']
        roles = target.get('role_signature')
        try:
            if (program_ref.get('version') != reference['version']
                    or _path(program_ref) != expected_path
                    or _absolute_reference(program_ref, self.workspace).get('file') != str(
                        self.workspace / 'research_state.json')
                    or not isinstance(roles, dict)
                    or any(theory.get(key) != target.get(key) for key in ('output_unit', 'context_ref'))
                    or any(roles.get(name) != contract
                           for name, contract in theory.get('role_signature', {}).items())):
                return None
            # Use the live index, which may be newer than the last disk request.
            value, = resolve_program_sources(self, [{'file': 'research_state.json',
                                                     'json_path': expected_path}])
            program = Program.from_dict(value)
            if program != Program.from_dict(theory['program']):
                return None
            declared = theory.get('role_signature', {})
            if any(not isinstance(declared.get(name), dict)
                   or any(declared[name].get(key) is None for key in ('role', 'unit', 'time'))
                   for name in program.required_features()):
                return None
            expected = export_temporal_program(program, feature_names=tuple(roles)).encode('utf-8')
            if (self.workspace / self._evaluated_summary()['model']).read_bytes() != expected:
                return None
        except (OSError, KeyError, TypeError, ValueError):
            return None
        return reference['id']

    def _bind_revision_target(self, question):
        """Bind a funded revision to a retained Program or a measured research before.

        A bank's fitting champion is never relabeled as an adopted predictor.
        The fallback research before uses only its declared registration order.
        Paid rows remain at their live observation addresses, including rows
        acquired since the last request/checkpoint was written.
        """
        target = question.get('theory_target') or {}
        if target.get('revision_fitting') is None:
            return
        automatic_parent = target.get('parent_selection_ref') is not None
        if not target.get('parent_theory_id') or automatic_parent:
            target.pop('parent_binding_role', None)
            target.pop('parent_selection_evidence', None)
            retained = self.evaluated_model or {}
            selected = retained.get('source_model')
            versions = _question_versions(self)
            revision_parent = self._retained_revision_parent(question, target)
            matches = [revision_parent] if revision_parent is not None else []
            if revision_parent is not None:
                target['parent_binding_role'] = 'retained_revision_program'
            for identifier, theory in self.theory_index.items():
                if revision_parent is not None:
                    break
                source = theory.get('source_ref', {})
                if source.get('kind') != 'native_candidate':
                    continue
                owner = versions.get((source['question_id'], source.get('meaning_version', 1)))
                if owner is None:
                    continue
                bank, owner_ref = owner
                rounds = bank.get('native_condition_research', [])
                r, i = source['round_index'], source['candidate_index']
                if not (0 <= r < len(rounds) and 0 <= i < len(rounds[r].get('candidates', []))):
                    continue
                candidate = rounds[r]['candidates'][i]
                reference = theory.get('program_source_ref', {})
                tail = ['native_condition_research', r, 'candidates', i, 'program']
                # Source meanings may move to meaning_history; the original
                # canonical registration remains an address to that same bank.
                addresses = [[*_path(owner_ref), *tail], [*_path(owner_ref)[:2], *tail]]
                if (selected is not None and candidate.get('model') == selected
                        and candidate.get('program') is not None
                        and reference.get('file') == 'research_state.json'
                        and _path(reference) in addresses):
                    matches.append(identifier)
            if len(matches) != 1:
                from hegel_machine.cognition.retained_native_parent import bind_retained_native_parent

                historical = bind_retained_native_parent(self, question)
                if historical['status'] == 'bound':
                    matches = [historical['theory_id']]
                else:
                    research_parent = self._research_candidate_parent(question, target) if not matches else None
                    if research_parent is not None:
                        matches = [research_parent['theory_id']]
                        target['parent_binding_role'] = 'research_candidate_before'
                        target['parent_selection_evidence'] = research_parent
                    else:
                        target.pop('parent_theory_id', None)
                        target.pop('parent_selection_ref', None)
                        if target.pop('paid_evidence_scope', None) is not None:
                            target.pop('evidence_checks', None)
                        target['parent_binding_obligation'] = {
                            'type': 'AuthorWorkRequest', 'reason': 'unique_retained_native_parent_required',
                            'required': ['retained native Program or registered native candidate '
                                         'measured to fail on the complete current paid scope'],
                            'source_model': selected,
                            'source_ref': {'file': 'research_state.json', 'json_path': ['evaluated_model']},
                            'matching_theory_ids': matches,
                            'retained_native_reason': historical['reason']}
                        return
            target['parent_theory_id'] = matches[0]
            target['parent_selection_ref'] = (deepcopy(target['parent_selection_evidence']['fit_mse_ref'])
                if target.get('parent_binding_role') == 'research_candidate_before' else
                {'file': 'research_state.json', 'json_path': ['evaluated_model']})
            target.pop('parent_binding_obligation', None)
        # An author's explicit checks retain their own scope and semantics.
        if target.get('evidence_checks') and not target.get('paid_evidence_scope'):
            return
        n = len(self.rows)
        threshold = self.config.confirmation_mse_threshold
        criterion_ref = {'file': 'research_state.json',
                         'json_path': ['limits', 'confirmation_mse_threshold']}
        criterion = {'kind': 'paid_mse', 'evaluated_rows': n,
                     'mse_threshold': threshold, 'criterion_ref': criterion_ref}
        checks = [{'source_ref': {'file': 'laboratory_observations.json',
                    'json_path': [record_index, 'rows', row_index]},
                   'context_ref': deepcopy(target['context_ref']), 'criterion': deepcopy(criterion)}
            for record_index, record in enumerate(self.observations)
            for row_index, _ in enumerate(record.get('rows', []))]
        if not n or len(checks) != n:
            return
        target['evidence_checks'] = checks
        target['paid_evidence_scope'] = {'records': len(self.observations), 'rows': n,
            'criterion_ref': criterion_ref, 'scope': 'current_complete_paid_prefix_not_confirmation'}

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

    def _outgoing_files_for_result(self, status, manifest, manifest_paths, model_paths):
        # Registered studies publish the existing workspace result protocol.
        # Reading every predictor here only builds a mapping discarded by
        # compact_workspace_result; the complete independent files stay on disk.
        return {}

    def result(self, status):
        result = super().result(status)
        result['research_organization'] = deepcopy(self._organization_state)
        return compact_workspace_result(self, result)

    def checkpoint_result(self, status):
        result = super().checkpoint_result(status)
        result['research_organization'] = deepcopy(self._organization_state)
        return compact_workspace_result(self, result)

    def _result_files(self, manifest, manifest_paths, model_paths):
        # The existing method_practice_workspace protocol owns this result's
        # files on disk; its loader uses Common's full archive exporter.
        return {}


def _source_cohort_allocation(plan):
    """Reserve final C aggregation inside the registered sources' original caps."""
    organization = plan.get('research_organization') or {}
    count = len(plan['cases'])
    if organization.get('phase') != 'source' or count < 2:
        return None
    work = organization.get('consolidation', {})
    mission = work.get('mission', {})
    funding = mission.get('allocation', mission.get('theory_learning_allocation', 0))
    funding = funding.get('compute', 0) if isinstance(funding, dict) else funding
    requested = work.get('allocation', {}).get('compute', funding)
    if any(type(value) is not int or value < 0 for value in (funding, requested)):
        raise ValueError('source cohort compute allocation needs nonnegative integer compute')
    cap = plan['limits']['compute_budget']
    reserved = min(requested, cap * count, funding * count)
    share, remainder = divmod(reserved, count)
    return {'source_compute_limit': cap * count, 'aggregation_compute': reserved,
        'source_stage_compute': {case['id']: cap - share - int(index < remainder)
                                 for index, case in enumerate(plan['cases'])},
        'local_consolidation_compute': 0}


def _prepare_organization(plan, output, knowledge, previous_study=None):
    """Anchor the small registered libraries once, keeping all old knowledge."""
    arms = set(plan['arms'])
    if not arms.intersection(ORGANIZATION_ARMS):
        return None, resolve_workspace_archive(knowledge) if knowledge is not None else None
    if arms != set(ORGANIZATION_ARMS):
        raise ValueError('register archive, fixed_pr2 and pr3 together')
    registration = deepcopy(plan.get('research_organization') or {})
    validate_exploration_policy(registration.get('exploration_policy'))
    if registration.get('phase') not in {'source', 'target'}:
        raise ValueError('three-arm organization needs a registered source or target phase')
    if (plan.get('source_cohort_allocation') is not None
            and plan['source_cohort_allocation'] != _source_cohort_allocation(plan)):
        raise ValueError('source_cohort_allocation must partition the registered original caps')
    for case in plan['cases']:
        if 'arms' in case and set(case['arms']) != arms:
            raise ValueError('each source must retain all three registered organization groups')
    source_work = registration.get('consolidation', {})
    if registration['phase'] == 'source':
        for reference in source_work.get('source_question_refs', []):
            if (not isinstance(reference, dict) or not isinstance(reference.get('question_id'), str)
                    or not reference['question_id'] or type(reference.get('meaning_version')) is not int
                    or reference['meaning_version'] < 1):
                raise ValueError('source_question_refs require explicit question_id and meaning_version')
        mission = source_work.get('mission', {})
        allowance = mission.get('theory_learning_allocation', mission.get('allocation', {}))
        allowance = allowance.get('compute') if isinstance(allowance, dict) else allowance
        if (mission.get('purpose') not in {'theory_learning', 'method_learning'}
                or not mission.get('mission_id') or type(allowance) is not int
                or not 0 <= allowance <= plan['limits']['compute_budget']):
            raise ValueError('source organization needs one registered bounded theory mission')
        if source_work.get('theories', registration.get('theories', [])) != registration.get('theories', []):
            raise ValueError('registered source declarations must be shared by all three groups')
    memory = resolve_workspace_archive(deepcopy(knowledge or {}))
    selected = registration.get('initial_library_path') or memory.get('theory_consolidation')
    if previous_study:
        original_input = json.loads((previous_study / 'initial_archive.json').read_text())
        memory = deepcopy(original_input)
        selected = original_input['theory_consolidation']
    fixed = output / 'initial_theory_library.json'
    if selected:
        reference = ({'file': str(Path(selected).resolve()), 'json_path': []}
                     if isinstance(selected, str) else deepcopy(selected))
        # A prepared common input already names this output's frozen library.
        # Preserve its original source instead of replacing it with itself.
        if not (Path(reference['file']).resolve() == fixed.resolve()
                and not _path(reference)):
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


def _validate_finite_author(author, organization):
    if author is None:
        return
    if (not organization or not isinstance(author, dict) or author.get('kind') != 'finite_native'
            or not isinstance(author.get('path'), str) or not author['path']
            or type(author.get('max_steps')) is not int or author['max_steps'] < 1
            or type(author.get('confirm')) is not bool):
        raise ValueError('finite author needs registered ABC, kind finite_native, path, max_steps and explicit confirm')


def protocol(*, kinds=("memoryless", "delay", "hysteresis"), cases=1,
             arms=("code_react", "hegel"), seed=61011017, config=None,
             model="gpt-6-luna", timeout_seconds=300, archive=None,
             research_organization=None, author=None):
    registered = set(arms).intersection(ORGANIZATION_ARMS)
    if registered and set(arms) != set(ORGANIZATION_ARMS):
        raise ValueError('register archive, fixed_pr2 and pr3 together')
    if bool(registered) != (research_organization is not None):
        raise ValueError('registered arms require research_organization; legacy arms do not use it')
    if research_organization is not None and research_organization.get('phase') not in {'source', 'target'}:
        raise ValueError('three-arm organization needs a registered source or target phase')
    if research_organization is not None:
        validate_exploration_policy(research_organization.get('exploration_policy'))
    _validate_finite_author(author, research_organization)
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
        plan['study'] = 'anonymous_registered_question_theory_research'
        plan['controls'] = ('All three groups use Common code_react, the same public task root, '
            'complete original archive, tools, budgets and terminal order. B/C automatically '
            'route actual task contracts through PR2. Only C consolidates registered source '
            'evidence. Fixed libraries constrain automatic organization; ordinary Python, '
            'explicit research tools and authored candidates remain available to every group. '
            'Target automatic libraries remain frozen. Multi-source C stages reserve their '
            'registered cohort-consolidation share inside the unchanged total source allowance.')
        cohort = _source_cohort_allocation(plan)
        if cohort is not None:
            plan['source_cohort_allocation'] = cohort
    if author is not None:
        plan['author'] = deepcopy(author)
        plan['model'] = None
        plan['provider'] = {'backend': 'finite_native_procedure', 'model_calls': 0}
        plan['runtime_snapshot'] = sorted({str(path.relative_to(ROOT))
            for path in (ROOT / 'src/hegel_machine').rglob('*.py')} |
            {'scripts/run_anonymous_revision_inquiry.py', 'scripts/run_online_inquiry_llm.py',
             'scripts/run_online_inquiry_study.py'})
        plan['runtime_entry'] = str(ROOT / 'scripts/run_anonymous_revision_inquiry.py')
        plan['author_source_ref'] = {'file': 'initial_archive.json',
                                     'json_path': ['files', author['path']]}
        plan['discovery'] = ('The preregistered finite procedure runs through the same Common tools and complete '
            'Python archive in every group. Its supplied algorithms and any supplied scientific content remain '
            'identified as inputs; only actually generated and checked objects are outputs. No model is called.')
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


def _aggregate_registered_sources(plan, output, trials, organization, knowledge,
                                  paid_traces, previous_study=None):
    """Consolidate C's actual sources inside their original combined allowance."""
    sources = [trial['result'] for trial in trials if trial['arm'] == 'pr3']
    references = [{'file': str(Path(source['workspace']) / 'training.json'), 'json_path': [],
        'source_theory_ids': [entry['id'] for entry in source['research_organization'].get(
            'consolidation_basis', {}).get('source_versions', [])]} for source in sources]
    source_versions = _episode_file_versions(SimpleNamespace(workspace=Path(output)),
        [{'file': str(Path(source['workspace']) / name)} for source in sources
         for name in ('training.json', 'research_state.json')])
    source_costs = {key: (None if any(source['costs'].get(key) is None for source in sources)
        else sum(source['costs'][key] for source in sources)) for key in sources[0]['costs']}
    previous = None
    if previous_study and (Path(previous_study) / 'source_aggregation.json').is_file():
        previous = json.loads((Path(previous_study) / 'source_aggregation.json').read_text())
        snapshot = Path(previous_study) / 'source_at_run'
        runtime_unchanged = (not snapshot.is_dir() or all(
            (snapshot / name).is_file() and (ROOT / name).is_file()
            and (snapshot / name).read_bytes() == (ROOT / name).read_bytes()
            for name in plan.get('runtime_snapshot', [])))
        if (previous['source_refs'] == references and previous['source_costs'] == source_costs
                and previous['source_versions'] == source_versions and runtime_unchanged):
            retained = {**previous, 'new_compute_cost': 0, 'reused': True}
            write_json(Path(output) / 'source_aggregation.json', retained)
            return retained
    prior_compute = previous['cumulative_aggregation_compute'] if previous else 0
    prior_mission = previous['cumulative_aggregation_mission_compute'] if previous else 0
    mission = deepcopy(organization['consolidation']['mission'])
    funding_key = 'allocation' if 'allocation' in mission else 'theory_learning_allocation'
    funding = mission[funding_key]
    total_funding = (funding['compute'] if isinstance(funding, dict) else funding) * len(sources)
    mission[funding_key] = ({**funding, 'compute': total_funding}
                           if isinstance(funding, dict) else total_funding)
    mission_before = sum(source['research_organization']['theory_mission'].get(
        'resource_usage', {}).get('compute', 0) for source in sources) + prior_mission
    mission['resource_usage'] = {'compute': mission_before}
    config = InquiryLLMConfig(**{**plan['limits'],
        'compute_budget': plan['limits']['compute_budget'] * len(sources),
        'max_model_calls': 0, 'token_budget': 0})
    work = deepcopy(organization['consolidation'])
    for key in ('source_theory_ids', 'source_question_refs', 'episode_sources'):
        work.pop(key, None)
    work.update(common_source_refs=references, mission=mission)
    memory = deepcopy(knowledge or {})
    if previous:
        memory['theory_consolidation'] = {'file': previous['library_path'], 'json_path': []}
        memory['theory_index'] = {'file': previous['index_path'], 'json_path': []}
    workspace = Path(output) / 'source_aggregation'
    agent = RegisteredOrganizationAgent(None, workspace, arm='pr3',
        research_organization={**deepcopy(organization), 'consolidation': work},
        knowledge=memory, config=config, source_task=False, paid_trace_references=paid_traces)
    # Earlier per-case research and earlier aggregate passes remain spent. The
    # ordinary operator clips this original balance and the mission balance.
    compute_before = source_costs['compute_cost'] + prior_compute
    agent.costs['compute_cost'] = compute_before
    owner_id = 'organization:source:' + mission['mission_id']
    owner = agent._open_question({'id': owner_id, 'queue': 'theory', 'parent_id': None,
        'question': 'Consolidate the registered physical sources, including failed outcomes.',
        'mission': mission})
    agent._automatic_work({**work, 'operator': 'theory_consolidation',
        'research_question': {'id': owner_id}}, None, 'cohort_consolidation')
    feedback = deepcopy(agent.events[-1]['feedback'])
    actual = agent.costs['compute_cost'] - compute_before
    mission_actual = owner['mission'].get('resource_usage', {}).get('compute', 0) - mission_before
    agent._organization_state['theory_mission'] = deepcopy(owner['mission'])
    result = agent.result(feedback['status'])
    # No new object is a valid source outcome. Keep its advertised empty index
    # address readable by the same frozen target installer.
    save_theory_index(agent)
    write_json(workspace / 'training.json', result)
    write_json(workspace / 'archive.json', result['outgoing_knowledge'])
    receipt = {'status': feedback['status'], 'source_refs': references,
        'source_versions': source_versions, 'source_costs': source_costs,
        'source_compute_limit': config.compute_budget,
        'source_compute_before': compute_before, 'mission_compute_limit': total_funding,
        'mission_compute_before': mission_before, 'new_compute_cost': actual,
        'cumulative_aggregation_compute': prior_compute + actual,
        'cumulative_aggregation_mission_compute': prior_mission + mission_actual,
        'source_compute_after': agent.costs['compute_cost'],
        'mission_compute_after': owner['mission'].get('resource_usage', {}).get('compute', 0),
        'library_path': str(workspace / LIBRARY_PATH),
        'index_path': str(workspace / 'shared/theory_index.json'),
        'archive': str(workspace / 'archive.json'), 'feedback': feedback, 'reused': False,
        'new_observations': 0, 'new_model_calls': 0}
    write_json(Path(output) / 'source_aggregation.json', receipt)
    return receipt


def run_study(plan, output, *, provider_factory=None, knowledge=None, prepare_prefix=None,
              laboratory_factory=None, workers=1, agent_factory=None,
              previous_study=None, resume_task=''):
    return _run_study(plan, output, provider_factory=provider_factory, knowledge=knowledge,
        prepare_prefix=prepare_prefix, laboratory_factory=laboratory_factory, workers=workers,
        agent_factory=agent_factory, previous_study=previous_study, resume_task=resume_task)


def run_target_intervention(plan, output, *, keep_study, prepare_prefix=None,
                            laboratory_factory=None, workers=1, agent_factory=None):
    """Run only the removed-library C jobs, paired with completed frozen ABC.

    The complete ABC plan and its checks stay intact. The prior run supplies
    common initial knowledge, never the target training state or terminal rows.
    All fresh training, confirmation and evaluation use the ordinary study loop.
    """
    from hegel_machine.cognition.theory_consolidation import _read_library

    keep = Path(keep_study).resolve()
    original = json.loads((keep / 'protocol.json').read_text())
    summary = json.loads((keep / 'summary.json').read_text())
    if (set(plan['arms']) != set(ORGANIZATION_ARMS)
            or plan.get('research_organization', {}).get('phase') != 'target'
            or original.get('research_organization', {}).get('phase') != 'target'
            or summary['status'] != 'completed' or not plan.get('author')):
        raise ValueError('target intervention requires a completed finite ABC target study')
    for field in ('cases', 'arms', 'model', 'provider', 'limits', 'prefix_rows',
                  'evaluation', 'author', 'archive', 'prepared_public_questions', 'source_cost_ref',
                  'paid_score_index_refs_by_case'):
        if plan.get(field) != original.get(field):
            raise ValueError(f'target intervention must preserve registered {field}')
    organization = deepcopy(plan['research_organization'])
    retained_organization = deepcopy(original['research_organization'])
    removed_library = organization.pop('pr3_library_path')
    retained_organization.pop('pr3_library_path')
    if organization != retained_organization:
        raise ValueError('only the automatic C library may change in a target intervention')
    expected = {(case['id'], arm) for case in plan['cases'] for arm in ORGANIZATION_ARMS}
    trial_keys = [(trial['case'], trial['arm']) for trial in summary['trials']]
    if len(trial_keys) != len(expected) or set(trial_keys) != expected:
        raise ValueError('the keep study must retain every registered ABC outcome, including failures')
    runtime = sorted({str(path.relative_to(ROOT))
        for path in (ROOT / 'src/hegel_machine').rglob('*.py')} |
        {'scripts/run_anonymous_revision_inquiry.py'} | set(plan.get('runtime_snapshot', [])))
    if original.get('runtime_snapshot') != runtime or any(
            (ROOT / name).read_bytes() != (keep / 'source_at_run' / name).read_bytes()
            for name in runtime):
        raise ValueError('paired target intervention requires the same current frozen runtime')
    intervention = deepcopy(plan.get('target_intervention') or {})
    if intervention.get('kind') not in {'individual_object', 'new_redundancy_class'}:
        raise ValueError('declare the frozen individual-object or new-class intervention')
    reference = intervention['manifest_ref']
    manifest = json.loads(Path(reference['file']).read_text())
    for key in _path(reference):
        manifest = manifest[key]
    if manifest.get('status') != 'frozen_ready':
        raise ValueError('target intervention needs a ready frozen removal manifest')
    removed = set(manifest['removed_candidate_ids'])
    retained_library = _read_library({'file': str(keep / 'pr3_theory_library.json'), 'json_path': []}, ROOT)
    selected_library = _read_library({'file': removed_library, 'json_path': []}
        if isinstance(removed_library, str) else removed_library, ROOT)
    old = organization['initial_library_path']
    old_library = _read_library({'file': old, 'json_path': []} if isinstance(old, str) else old, ROOT)
    prior = retained_library.get('candidates', {})
    actual = selected_library.get('candidates', {})
    old_ids = set(old_library.get('candidates', {}))
    if (not removed or not removed <= set(prior) or removed.intersection(old_ids)
            or not old_ids <= set(manifest['preserved_old_candidate_ids'])
            or list(actual) != [key for key in prior if key not in removed]
            or actual != {key: value for key, value in prior.items() if key not in removed}):
        raise ValueError('intervention must remove only the declared new candidates and preserve the complete old library')
    knowledge = json.loads((keep / 'initial_archive.json').read_text())
    intervention.update(keep_study=str(keep), executed_arms=['pr3'],
        keep_protocol_ref={'file': str(keep / 'protocol.json'), 'json_path': []},
        keep_summary_ref={'file': str(keep / 'summary.json'), 'json_path': []},
        keep_outcome_refs=[{'file': str(keep / 'summary.json'), 'json_path': ['trials', index]}
                           for index in range(len(summary['trials']))],
        common_initial_archive_ref={'file': str(keep / 'initial_archive.json'), 'json_path': []},
        keep_costs_recharged=False)
    return _run_study(plan, output, knowledge=knowledge, prepare_prefix=prepare_prefix,
        laboratory_factory=laboratory_factory, workers=workers, agent_factory=agent_factory,
        target_intervention=intervention)


def _run_study(plan, output, *, provider_factory=None, knowledge=None, prepare_prefix=None,
               laboratory_factory=None, workers=1, agent_factory=None,
               previous_study=None, resume_task='', target_intervention=None):
    output = Path(output).resolve()
    author = plan.get('author')
    _validate_finite_author(author, plan.get('research_organization'))
    if author is None and provider_factory is None:
        raise ValueError('model-authored study requires provider_factory')
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
        if plan.get('source_cohort_allocation') != previous_plan.get('source_cohort_allocation'):
            raise ValueError('cohort recovery must preserve registered source_cohort_allocation')
        if author != previous_plan.get('author'):
            raise ValueError('cohort recovery must preserve registered author')
    organization, prepared_knowledge = _prepare_organization(plan, output, knowledge,
        Path(target_intervention['keep_study']) if target_intervention else previous_study)
    if author:
        if author['path'] not in (prepared_knowledge or {}).get('files', {}):
            raise ValueError('finite author path must exist in the common executable archive')
        # Freeze the actual current runtime once for the whole study, never as
        # an import path. The supplied author text is saved in initial_archive.
        plan = deepcopy(plan)
        plan['runtime_snapshot'] = sorted({str(path.relative_to(ROOT))
            for path in (ROOT / 'src/hegel_machine').rglob('*.py')} |
            {'scripts/run_anonymous_revision_inquiry.py', 'scripts/run_online_inquiry_llm.py',
             'scripts/run_online_inquiry_study.py'} | set(plan.get('runtime_snapshot', [])))
        plan['runtime_entry'] = str(ROOT / 'scripts/run_anonymous_revision_inquiry.py')
        plan['author_source_ref'] = {'file': 'initial_archive.json',
                                     'json_path': ['files', author['path']]}
        for name in plan['runtime_snapshot']:
            path = output / 'source_at_run' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            copy2(ROOT / name, path)
    agent_factory = agent_factory or (RegisteredOrganizationAgent if organization else CommonArchiveInquiryAgent)
    write_json(output / "protocol.json", plan)
    # Literal common input is saved once; no arm learns the other's live results.
    if target_intervention:
        clone_or_copy(Path(target_intervention['common_initial_archive_ref']['file']),
                      output / 'initial_archive.json')
    else:
        write_json(output / "initial_archive.json", prepared_knowledge or {})
    source_workspace = (prepared_knowledge or {}).get('workspace')
    if source_workspace is None and plan.get('archive'):
        source_workspace = Path(plan['archive']).resolve().parent
    if previous_study and (previous_study / 'reference_archive').is_dir():
        source_workspace = previous_study / 'reference_archive'
    if target_intervention and (Path(target_intervention['keep_study']) / 'reference_archive').is_dir():
        source_workspace = Path(target_intervention['keep_study']) / 'reference_archive'
    paid_traces = archived_paid_trace_references((prepared_knowledge or {}).get('files', {}),
        output / 'reference_archive', source_workspace=source_workspace)
    limits = InquiryLLMConfig(**plan["limits"])
    if laboratory_factory is None:
        laboratory_factory = lambda case: AnonymousHistoryLab(kind=case["kind"], seed=case["seed"])
    jobs = []
    for case_index, case in enumerate(plan["cases"]):
        arms = plan["arms"]
        ordered = case.get("arms", arms[case_index % len(arms):] + arms[:case_index % len(arms)])
        jobs.extend((case, arm) for arm in ordered
                    if not target_intervention or arm == 'pr3')

    def train(job):
        case, arm = job
        workspace = output / case["id"] / arm
        old_workspace = previous_study / case['id'] / arm if previous_study else None
        if old_workspace and (old_workspace / 'training.json').is_file():
            result = json.loads((old_workspace / 'training.json').read_text())
            return {'case': case, 'arm': arm, 'result': result, 'initial_event_count': 0}
        job_limits, job_organization = limits, organization
        if arm == 'pr3' and plan.get('source_cohort_allocation'):
            cohort = plan['source_cohort_allocation']
            job_limits = replace(limits, compute_budget=cohort['source_stage_compute'][case['id']])
            job_organization = deepcopy(organization)
            job_organization['consolidation'].setdefault('allocation', {})['compute'] = (
                cohort['local_consolidation_compute'])
        agent = agent_factory(None if author else provider_factory(workspace), workspace,
            arm=arm, knowledge=deepcopy(prepared_knowledge or {}), config=job_limits,
            confirmation_seed=case["confirmation_seed"], source_task=False if organization else not bool(knowledge),
            paid_trace_references=paid_traces,
            **({'research_organization': job_organization} if organization else {}))
        if author:
            agent._organization_state['finite_author'] = {'contract': deepcopy(author), 'completed_steps': 0}
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
        checkpoint = lambda data, path=workspace: write_json(path / 'active.json', data)
        if author:
            state = agent._organization_state['finite_author']

            def after_step(lab):
                state['completed_steps'] += 1
                agent._after_model_turn(lab)

            remaining = max(0, author['max_steps']-state['completed_steps'])
            outcome = (execute_procedure(agent, laboratory, author['path'], max_steps=remaining,
                allow_confirmation=author['confirm'], record_events=True, on_checkpoint=checkpoint,
                before_step=agent._before_model_turn, after_step=after_step)
                if remaining else {'status': 'method_step_limit'})
            result = agent.result(outcome['status'])
            checkpoint(result)
        else:
            result = agent.run(laboratory, on_checkpoint=checkpoint)
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
    elif author:
        trials = _native_process_trials(train, jobs, workers)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # map retains registration order; all training finishes before any
            # terminal rows are generated by the separate evaluation loop.
            trials = list(executor.map(train, jobs))

    aggregation = None
    if organization and organization['phase'] == 'source' and len(plan['cases']) > 1:
        aggregation = _aggregate_registered_sources(plan, output, trials, organization,
            prepared_knowledge, paid_traces, previous_study)
    records = []
    for trial in trials:
        case, result = trial["case"], trial["result"]
        host_workspace = output / case['id'] / trial['arm']
        host_workspace.mkdir(parents=True, exist_ok=True)
        evaluation = {"status": "no_submitted_model", "mse": None,
                      "evaluation_rows": 0, "python_execution_seconds": 0.}
        if result["model"]:
            rows = laboratory_factory(case).evaluation_rows(
                length=plan["evaluation"]["rows"], seed=case["evaluation_seed"])
            evaluation = evaluate_terminal_model(result["workspace"], result["model"], rows,
                                                 timeout_seconds=limits.execution_seconds)
            predictions = evaluation.pop("predictions", None)
            if predictions is not None:
                training_ref = {'file': str(Path(result['workspace']) / 'training.json'), 'json_path': []}
                prediction_path = host_workspace / 'terminal_predictions.json'
                write_json(prediction_path, {'model': result['model'], 'training_ref': training_ref,
                    'model_ref': {**training_ref, 'json_path': ['model']},
                    'submitted_model_ref': {'file': str(Path(result['workspace']) / result['model'])},
                    'evaluation_contract': {'case': case['id'], 'kind': case['kind'],
                        'evaluation_seed': case['evaluation_seed'], 'rows': plan['evaluation']['rows'],
                        'trajectory_length': plan['evaluation'].get('trajectory_length')},
                    'evaluation_contract_ref': {'file': str(output / 'protocol.json'), 'json_path': ['evaluation']},
                    'predictions': predictions})
                evaluation['prediction_ref'] = {'file': str(prediction_path), 'json_path': []}
        evaluation["passed"] = (evaluation.get("returncode") == 0 and
            evaluation.get("mse", float("inf")) <= plan["evaluation"]["mse_threshold"])
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
               "evidence_kind": 'finite_native_program_execution' if author else "real_external_model" if live_events and all(
                   event.get("usage", {}).get("backend") == "codex_cli" for event in live_events)
                   else "scripted_interface_check",
               "source_count": len(plan["cases"]), "scope": plan["scope"]}
    if target_intervention:
        summary['target_intervention'] = deepcopy(target_intervention)
    if aggregation is not None:
        summary['source_aggregation'] = aggregation
    if plan.get('source_cohort_allocation') is not None:
        summary['source_cohort_allocation'] = deepcopy(plan['source_cohort_allocation'])
    write_json(output / "summary.json", summary)
    if organization and organization['phase'] == 'target':
        from hegel_machine.cognition.method_effect_query import registered_successor_effect

        try:
            effect = (registered_successor_effect(target_intervention['keep_study'], output,
                left_arm='pr3', right_arm='pr3', source_cost_ref=plan.get('source_cost_ref'),
                intervention_ref=target_intervention['manifest_ref']) if target_intervention else
                registered_successor_effect(output, source_cost_ref=plan.get('source_cost_ref')))
        except (OSError, KeyError, ValueError) as error:
            effect = {'status': 'UNKNOWN', 'reason': f'{type(error).__name__}: {error}',
                'training_and_evaluation_retained': True}
        write_json(output / 'research_effects.json', effect)
        summary['research_effects_ref'] = {'file': str(output / 'research_effects.json'), 'json_path': []}
        write_json(output / 'summary.json', summary)
    return summary


def evaluate_terminal_model(workspace, model, rows, *, timeout_seconds):
    """Execute an already submitted model once on evaluator-only rows.

    The shared Python executor projects away targets before prediction. Its
    existing optional source-fit behavior is unchanged; callers requiring a
    literal frozen predictor must verify that execution context before calling.
    """
    started = monotonic()
    evaluation = evaluate_python_model(workspace, model, rows,
                                       timeout_seconds=timeout_seconds)
    evaluation.update(python_execution_seconds=monotonic() - started,
                      evaluation_rows=len(rows))
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", choices=("memoryless", "delay", "hysteresis"),
                        default=["memoryless", "delay", "hysteresis"])
    parser.add_argument("--cases", type=int, default=1, help="Independent source seeds per control family")
    parser.add_argument("--arms", nargs="+", choices=("code_react", "hegel", *ORGANIZATION_ARMS), required=True)
    parser.add_argument("--research-organization", type=Path,
                        help="Registered source/target organization JSON for archive fixed_pr2 pr3")
    parser.add_argument('--finite-native-author', type=Path,
                        help='Explicit finite author JSON: kind, shared path, max_steps and confirm')
    parser.add_argument("--archive", type=Path, help="Identical incoming executable archive for all arms")
    parser.add_argument("--seed", type=int, default=61011017)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--timeout-seconds", type=float, default=300.)
    parser.add_argument("--workers", type=int,
        help="Native jobs use independent CPU processes (default 8); model authors retain threads (default 1)")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    organization = json.loads(args.research_organization.read_text()) if args.research_organization else None
    author = json.loads(args.finite_native_author.read_text()) if args.finite_native_author else None
    try:
        plan = protocol(kinds=args.kinds, cases=args.cases, arms=args.arms, seed=args.seed,
                        model=args.model, timeout_seconds=args.timeout_seconds, archive=args.archive,
                        research_organization=organization, author=author)
    except ValueError as error:
        parser.error(str(error))
    if author is None:
        plan["provider"]["auth_mode"] = verify_codex_subscription()
    knowledge = json.loads(args.archive.read_text()) if args.archive else None
    def provider(workspace):
        return CodexCodeProvider(workspace, model=args.model, reasoning_effort="medium",
                                 timeout_seconds=args.timeout_seconds)
    print(json.dumps(run_study(plan, args.output, provider_factory=provider if author is None else None,
                               knowledge=knowledge,
                               workers=args.workers if args.workers is not None else 8 if author else 1), indent=2))


if __name__ == "__main__":
    main()
