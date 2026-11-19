"""Use unmodified canonicalizer AST definitions; no project dependencies required."""

from _common import HEGEL_REPO, AUDIT_OUTPUT_DIR
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

source = HEGEL_REPO / 'src/hegel_machine/cognition/open_workspace_wormhole.py'
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {'_SkeletonNormalizer', 'canonicalize_python_model_ast'}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'))

def digest(s):
    return canonicalize_python_model_ast(s)

same_annotation_different_code = [
    "_CORE_LAW_EXPR = \"f['x'] / (0.28 * f['x'] + 1.6)\"\ndef predict(row):\n    return row['features']['x']\n",
    "_CORE_LAW_EXPR = \"f['x'] / (0.28 * f['x'] + 1.6)\"\ndef predict(row):\n    return 100000.0\n",
]
different_laws_without_annotation = [
    "def predict(row):\n    f = row['features']\n    latent = f['x'] / (0.28 * f['x'] + 1.6)\n    return 0.0 + (1.0) * (1.0) * latent\n",
    "def predict(row):\n    f = row['features']\n    latent = f['x'] ** 5 + f['y']\n    return 0.0 + (1.0) * (1.0) * latent\n",
]
out = {
    'same_annotation_different_code': [digest(s) for s in same_annotation_different_code],
    'different_laws_without_annotation': [digest(s) for s in different_laws_without_annotation],
    'negative_numeric_constants': [digest("_CORE_LAW_EXPR = \"" + s + "\"\n") for s in ["f['x'] ** 2", "f['x'] ** -2"]],
}
for key, items in out.items():
    out[key] = {'digests_equal': items[0]['ast_skeleton_digest'] == items[1]['ast_skeleton_digest'], 'canonicalized': items}
(AUDIT_OUTPUT_DIR / 'ast_probe_result.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, indent=2))
