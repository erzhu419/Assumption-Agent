"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04_r00", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["node_00"], "node_03": ["__input__", "node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "0283c3bf450a3e7b3373497e2c165719b103d7aea9bc1aafb7d0441dbeb664cc", "node_01": "3a429e7f39c801b3746be5ef66fc874a86e98300b9e067ee21b75ff6f53eb499", "node_02": "14c9cc4fdef05fbf2fecd63ac6aae911c57141a41dffaf8d37b6b7a1f1f24c28", "node_03": "a8d5622b3718656b728350b9dd13b2cf9c3b92923315467be85c7df0a5594486"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _calibrate(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    mask = np.ones(len(u), dtype=bool)\\n    for _ in range(8):\\n        a, b = u[mask], v[mask]\\n        ac, bc = a-a.mean(0), b-b.mean(0)\\n        scale = (ac*bc).sum(0)/(ac*ac).sum(0)\\n        shift = b.mean(0)-a.mean(0)*scale\\n        r = np.linalg.norm(u*scale+shift-v, axis=1)\\n        mask = r <= max(5*np.median(r), .0006)\\n    return shift, scale\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    offset, scale = _calibrate(study[\'panel_a\'])\\n    raw = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    x = (raw-offset)/scale\\n    c = study[\'panel_c\']\\n    return {\'x\': x, \'u\': c[\'u\'], \'v1\': c[\'v1\'], \'v2\': c[\'v2\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _basis(x):\\n    return np.column_stack((np.ones(len(x)), *(x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i + 1, x.shape[1]))))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    panel = inputs[\'node_00\']\\n    coef = np.linalg.lstsq(_basis(panel[\'u\']), panel[\'v1\'], rcond=None)[0]\\n    return {\'b\': _basis(panel[\'x\']) @ coef}\\n", "node_02": "import numpy as np\\n\\n\\ndef _basis(x):\\n    return np.column_stack((np.ones(len(x)), x, np.cos(2.4 * x)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    panel = inputs[\'node_00\']\\n    coefficients = np.linalg.lstsq(_basis(panel[\'u\']), panel[\'v2\'], rcond=None)[0]\\n    return {\'d\': _basis(panel[\'x\']) @ coefficients}\\n", "node_03": "import numpy as np\\nfrom itertools import combinations_with_replacement\\n\\n\\ndef _readout_basis(x):\\n    mixed = np.column_stack([x[:, i]*x[:, j] for i, j in\\n                             combinations_with_replacement(range(x.shape[1]), 2)])\\n    return np.column_stack((np.ones(len(x)), x, .4*mixed))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    b = inputs[\'node_01\'][\'b\']\\n    d = inputs[\'node_02\'][\'d\']\\n    f = _readout_basis(np.column_stack((b, d)))\\n    n = len(train[\'x\'])\\n    a, xt = f[:n], f[n:]\\n    gram = a@a.T\\n    cross = xt@a.T\\n    eye = np.eye(n)\\n    yh = cross@np.linalg.solve(gram+.0003*eye, train[\'y\'])\\n    state = cross@np.linalg.solve(gram+.001*eye, train[\'traj\'])\\n    end = state[:, -1]\\n    return {\'y_hat\': yh, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}\\n"}'

def _canon_digest(obj) -> str:
    h = hashlib.sha256()
    def _feed(v):
        if v is None: h.update(b'N;')
        elif isinstance(v, bool): h.update(b'B1;' if v else b'B0;')
        elif isinstance(v, (int, np.integer)): h.update(f'I:{int(v)};'.encode())
        elif isinstance(v, (float, np.floating)):
            fv = float(v)
            if math.isnan(fv): h.update(b'F:NaN;')
            elif math.isinf(fv): h.update(b'F:Inf;')
            else: h.update(b'F:' + np.asarray([0.0 if fv == 0.0 else fv], dtype='<f8').tobytes() + b';')
        elif isinstance(v, str): h.update(f'S:{len(v)}:{v};'.encode())
        elif isinstance(v, np.ndarray):
            h.update(f'A:{v.dtype.kind}:{v.shape}:'.encode())
            if v.dtype.kind == 'f':
                a = np.asarray(v, dtype=np.float64)
                nm = np.isnan(a)
                cl = np.where(nm, 0.0, a)
                cl = np.where(cl == 0.0, 0.0, cl).astype('<f8')
                h.update(cl.tobytes('C'))
            else:
                h.update(np.asarray(v, dtype='<i8').tobytes('C'))
            h.update(b';')
        elif isinstance(v, dict):
            h.update(f'D:{len(v)}:'.encode())
            for k in sorted(v.keys(), key=str): _feed(str(k)); _feed(v[k])
            h.update(b';')
        elif isinstance(v, (list, tuple)):
            h.update(f'L:{len(v)}:'.encode())
            for it in v: _feed(it)
            h.update(b';')
        else:
            h.update(repr(v).encode())
    _feed(obj)
    return h.hexdigest()

def _load_node_fn(nid: str, src: str):
    ns = {'__name__': f'_hegel_node_{nid}'}
    exec(compile(src, f'<dag_node_{nid}>', 'exec'), ns)
    fn = ns.get('run_node') or ns.get('solve_batch')
    if not callable(fn):
        raise RuntimeError(f'DAG node {nid} missing callable run_node/solve_batch')
    return fn

_COMPILED_FNS = {nid: _load_node_fn(nid, src) for nid, src in json.loads(_NODE_SOURCES).items()}
_SPEC_OBJ = json.loads(_DAG_SPEC)

def solve_batch(study: dict) -> dict:
    outputs = {'__input__': deepcopy(study)}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        if len(_SPEC_OBJ['topo_order']) == 1 and srcs == ['__input__']:
            node_in = deepcopy(outputs['__input__'])
        else:
            node_in = {s: deepcopy(outputs[s]) for s in srcs}
        in_dig = _canon_digest(node_in)
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Anti-identity-shell check: intermediate node cannot merely echo its raw predecessor dict
            echoed = (out_dig == in_dig) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            if echoed:
                raise RuntimeError(f'DAG node {nid} failed anti-identity-shell check')
            non_identity_checks[nid] = True
        outputs[nid] = out_v
    trace_fp = os.environ.get('HEGEL_M101_DAG_TRACE_PATH')
    if trace_fp:
        trace_obj = {
            'dag_id': _SPEC_OBJ['dag_id'],
            'active_node_count': len(_SPEC_OBJ['topo_order']),
            'depth': _SPEC_OBJ['depth'],
            'max_fanout': _SPEC_OBJ['max_fanout'],
            'shared_fanout_nodes': _SPEC_OBJ['shared_fanout_nodes'],
            'topo_order': _SPEC_OBJ['topo_order'],
            'exec_counts': exec_counts,
            'all_exec_count_eq_1': all(v == 1 for v in exec_counts.values()),
            'non_identity_checks': non_identity_checks,
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
