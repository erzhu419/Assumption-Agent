"""External selectors use the exact host runtime, including frozen source roots."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hegel_machine.cognition.finite_native_author import export_finite_native_author
from hegel_machine.cognition.native_condition_research import native_structure_planning_contract
from hegel_machine.cognition.online_inquiry_llm import _PLAN_RUNNER
from hegel_machine.cognition.research_episodes import SEMANTIC_FIELDS
from hegel_machine.cognition.temporal_operator_agent import paid_python_evaluation_contract
from test_native_structural_discrimination import structural_scene


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = '''import importlib, runpy
def select_actions(spec, observations, memory):
    ordinary = runpy.run_path('ordinary.py')
    if memory['branch'] == 'paid_scope':
        returned = ordinary['_completed_choice']([], ['saved.py'], 12,
            memory['question'], {'questions': [memory['question']],
                'operator_results': [memory['report']]},
            paid_rows_view=observations[0]['rows'], public_spec=spec)
        assert returned['research_question_id'] == 'q'
        module = importlib.import_module('hegel_machine.cognition.paid_score_history')
    else:
        assert ordinary['_structure_planning_option_matches'](
            memory['option'], spec, observations, memory['question'])
        module = importlib.import_module('hegel_machine.cognition.native_condition_research')
    return {'matched': True, 'helper_origin': module.__file__}
'''


def selector_workspace(path, branch):
    path.mkdir()
    public, agent, _, question, previous, _ = structural_scene()
    planning = native_structure_planning_contract(public, agent.observations, previous)
    scope = {'rows': agent.observations[0]['rows'],
        'question': {key: question[key] for key in ('id', 'meaning_version', *SEMANTIC_FIELDS)
                     if key in question},
        'public_spec': public, 'evaluation_contract': paid_python_evaluation_contract()}
    (path / 'paid_scope.json').write_text(json.dumps(scope))
    report = {'operator': 'research_choice', 'arguments': {
        'operator': 'research_choice', 'models': ['saved.py']},
        'models_requested': 1, 'models_completed': 1, 'paid_rows': 12,
        'evaluations': [{'path': 'saved.py', 'paid_score_reuse': {
            'scope_ref': {'file': 'paid_scope.json', 'json_path': []}}}]}
    payload = {'spec': public, 'observations': agent.observations, 'memory': {
        'branch': branch, 'question': question, 'report': report,
        'option': {'round_index': previous['round_index'], 'proposal': {
            'action': 'run', 'arguments': {'operator': 'native_condition_research',
                'structure_planning': planning}}}}}
    original = export_finite_native_author()
    (path / 'ordinary.py').write_text(original)
    (path / 'selector.py').write_text(SELECTOR)
    return payload, original


@pytest.mark.parametrize('branch', ['paid_scope', 'structure_planning'])
def test_external_selector_imports_existing_helpers_without_pythonpath(tmp_path, branch):
    workspace = tmp_path / 'workspace'
    payload, original = selector_workspace(workspace, branch)
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, '-B', '-c', _PLAN_RUNNER,
        str(workspace / 'selector.py')], input=json.dumps(payload), cwd=workspace,
        env=environment, capture_output=True, text=True, timeout=30, check=True)
    result = json.loads(completed.stdout)
    assert result['matched'] is True
    assert Path(result['helper_origin']).resolve().is_relative_to(ROOT / 'src')
    assert (workspace / 'ordinary.py').read_text() == original


def test_isolated_frozen_runner_keeps_its_own_package_after_parent_path_is_removed(tmp_path):
    frozen = tmp_path / 'source_at_review'
    shutil.copytree(ROOT / 'src/hegel_machine', frozen / 'src/hegel_machine',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    workspace = tmp_path / 'workspace'
    payload, original = selector_workspace(workspace, 'paid_scope')
    # -I excludes the checkout and user site. Supply only declared third-party
    # dependencies, as in the existing registered snapshot import-closure test.
    dependencies = set()
    for name in ('numpy', 'scipy', 'cryptography', 'threadpoolctl'):
        origin = Path(importlib.util.find_spec(name).origin)
        dependencies.add(str(origin.parent.parent if origin.name == '__init__.py'
                             else origin.parent))
    bootstrap = '''import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
dependencies = json.loads(sys.argv[2])
selector = sys.argv[3]
sys.path[:0] = [str(root / 'src'), *dependencies]
from hegel_machine.cognition.online_inquiry_llm import _PLAN_RUNNER
# The real selector starts a new process without its parent's path/modules.
# Recreate that boundary before executing the exact newly frozen runner.
sys.path.remove(str(root / 'src'))
for name in list(sys.modules):
    if name == 'hegel_machine' or name.startswith('hegel_machine.'):
        del sys.modules[name]
sys.argv = ['selector', selector]
exec(_PLAN_RUNNER)
'''
    completed = subprocess.run([sys.executable, '-I', '-B', '-c', bootstrap,
        str(frozen), json.dumps(sorted(dependencies)), str(workspace / 'selector.py')],
        input=json.dumps(payload), cwd=workspace, capture_output=True, text=True,
        timeout=30, check=True)
    result = json.loads(completed.stdout)
    assert result['matched'] is True
    assert Path(result['helper_origin']).resolve().is_relative_to(frozen / 'src')
    assert (workspace / 'ordinary.py').read_text() == original
