"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "66e7106977184ba443e0198664690cccb586f16221bfe61000527b1157f95927", "node_01": "604e8aaf2f4d49e64817f8f32ff027513f63ab9c64f4d35582f0fe694d1483de"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _affine(u):\\n    return np.column_stack((np.ones(len(u)), u))\\n\\n\\ndef _fit_a(panel):\\n    a, v = _affine(panel[\'u\']), panel[\'v\']\\n    coef = np.linalg.lstsq(a, v, rcond=None)[0]\\n    # Fit in the forward direction; one weight covers all channels of a row.\\n    for _ in range(12):\\n        residual = np.linalg.norm(v - a @ coef, axis=1)\\n        scale = max(2.5 * np.median(residual), 1e-9)\\n        weight = 1 / np.hypot(1, residual / scale)\\n        coef = np.linalg.lstsq(a * weight[:, None], v * weight[:, None], rcond=None)[0]\\n    return coef\\n\\n\\ndef _v1_basis(u):\\n    a, b, c, d = u.T\\n    return np.column_stack((np.ones(len(u)), a*b, a*d, b*c, c*d))\\n\\n\\ndef _v2_basis(u):\\n    return np.column_stack((_affine(u), np.cos(2.4*u)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    x = np.concatenate((train[\'x\'], study[\'test\'][\'x\']))\\n    a = _fit_a(study[\'panel_a\'])\\n    u = np.linalg.solve(a[1:].T, (x - a[0]).T).T\\n    c = study[\'panel_c\']\\n    v1 = np.linalg.lstsq(_v1_basis(c[\'u\']), c[\'v1\'], rcond=None)[0]\\n    v2 = np.linalg.lstsq(_v2_basis(c[\'u\']), c[\'v2\'], rcond=None)[0]\\n    # The frozen quadratic readout acts on the two panel C responses.\\n    h = np.column_stack((_v1_basis(u) @ v1, _v2_basis(u) @ v2))\\n    return {\'h\': h, \'n_train\': len(train[\'x\']), \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_01": "import numpy as np\\nfrom itertools import combinations_with_replacement\\n\\n\\ndef _readout_basis(x):\\n    mixed = np.column_stack([x[:, i]*x[:, j] for i, j in\\n                             combinations_with_replacement(range(x.shape[1]), 2)])\\n    return np.column_stack((np.ones(len(x)), x, .4*mixed))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    f = _readout_basis(upstream[\'h\'])\\n    n = upstream[\'n_train\']\\n    a, xt = f[:n], f[n:]\\n    gram = a@a.T\\n    cross = xt@a.T\\n    eye = np.eye(n)\\n    yh = cross@np.linalg.solve(gram+.0003*eye, upstream[\'y\'])\\n    state = cross@np.linalg.solve(gram+.001*eye, upstream[\'traj\'])\\n    end = state[:, -1]\\n    return {\'y_hat\': yh, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}\\n"}'

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
