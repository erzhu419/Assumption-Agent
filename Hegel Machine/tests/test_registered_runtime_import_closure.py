"""A fresh source snapshot imports its real entries without the current checkout."""

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts import run_anonymous_revision_inquiry as study
from test_registered_native_author import plan

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {'scripts/run_anonymous_revision_inquiry.py',
           'scripts/prepare_pr3_source_successor.py',
           'scripts/run_online_inquiry_llm.py', 'scripts/run_online_inquiry_study.py',
           'scripts/run_method_action_pairing.py'}


def test_fresh_registered_snapshot_contains_script_import_closure(tmp_path, request):
    frozen = tmp_path / 'source_at_run'
    configured = plan(steps=1)
    declared = set(configured['runtime_snapshot']) | {'scripts/prepare_pr3_source_successor.py'}
    assert SCRIPTS <= declared
    # All actual scripts imports, including imports inside the two entrypoints,
    # must be satisfied by the same newly frozen script files.
    dependencies = set()
    for name in SCRIPTS:
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module == 'scripts':
                dependencies.update('scripts/' + alias.name + '.py' for alias in node.names)
            elif node.module.startswith('scripts.'):
                dependencies.add(node.module.replace('.', '/') + '.py')
    assert dependencies <= SCRIPTS
    for name in sorted(declared):
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    assert (frozen / 'scripts/run_online_inquiry_llm.py').read_text() == (
        ROOT / 'scripts/run_online_inquiry_llm.py').read_text()
    smoke = r'''
import importlib, json, pathlib, sys
snapshot = pathlib.Path(sys.argv[1]).resolve()
sys.path[:0] = [str(snapshot / 'src'), str(snapshot)]
# -I removes user-site packages as well as the checkout. Add only the current
# environment's declared third-party dependency roots, without processing .pth.
sys.path.extend(json.loads(sys.argv[2]))
modules = ['scripts.prepare_pr3_source_successor', 'scripts.run_anonymous_revision_inquiry',
           'scripts.run_online_inquiry_llm', 'scripts.run_online_inquiry_study',
           'scripts.run_method_action_pairing']
origins = {}
for name in modules:
    module = importlib.import_module(name)
    origin = pathlib.Path(module.__file__).resolve()
    assert origin.is_relative_to(snapshot), (name, str(origin))
    origins[name] = str(origin)
science_modules = [module for name, module in sys.modules.items()
                   if name == 'hegel_machine' or name.startswith('hegel_machine.')]
assert science_modules
assert all(pathlib.Path(module.__file__).resolve().is_relative_to(snapshot / 'src')
           for module in science_modules if getattr(module, '__file__', None))
writer = sys.modules['scripts.run_online_inquiry_llm'].write_json
assert pathlib.Path(writer.__code__.co_filename).resolve().is_relative_to(snapshot)
print(json.dumps({'status': 'isolated_snapshot_imports_complete', 'script_origins': origins,
    'loaded_science_module_count': len(science_modules), 'main_functions_called': 0,
    'experiments': 0, 'model_calls': 0}))
'''
    dependency_roots = set()
    for name in ('numpy', 'scipy', 'cryptography', 'threadpoolctl'):
        origin = Path(importlib.util.find_spec(name).origin)
        dependency_roots.add(str(origin.parent.parent if origin.name == '__init__.py' else origin.parent))
    completed = subprocess.run([sys.executable, '-I', '-B', '-c', smoke, str(frozen),
        json.dumps(sorted(dependency_roots))],
        cwd=tmp_path, check=True, capture_output=True, text=True)
    receipt = json.loads(completed.stdout)
    assert receipt['status'] == 'isolated_snapshot_imports_complete'
    path = tmp_path / 'smoke_receipt.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    request.node.user_properties.append(('smoke_receipt', str(path)))
