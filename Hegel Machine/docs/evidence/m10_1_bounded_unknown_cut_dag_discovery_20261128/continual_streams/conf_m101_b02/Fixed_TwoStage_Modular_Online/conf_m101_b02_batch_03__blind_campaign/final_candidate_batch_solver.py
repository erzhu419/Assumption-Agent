"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "8d9638a3b3a6fc656c1468438a1565a3159aaccb1d00760869bfe7ed974c6b0c"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates and fit the two response readouts.\\"\\"\\"\\n\\nimport numpy as np\\nfrom scipy.linalg import solve\\nfrom scipy.optimize import minimize_scalar, root\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _design(*blocks):\\n    return np.column_stack((np.ones(len(blocks[0])), *blocks))\\n\\n\\ndef _calibration(x):\\n    i, j = np.triu_indices(x.shape[1], 1)\\n    return _design(x, x*x, x*x*x, x[:, i]*x[:, j])\\n\\n\\ndef _coordinates(study, x):\\n    pa = study[\'panel_a\']\\n    f, v = _calibration(pa[\'u\']), pa[\'v\']\\n    w = _ls(f, v)\\n    for _ in range(60):\\n        q = np.maximum(np.linalg.norm(v-f@w, axis=1), 0.00025)**-0.5\\n        w = _ls(f*q[:, None], v*q[:, None])\\n    inside = np.linalg.norm(v-f@w, axis=1) < 0.005\\n    w = _ls(f[inside], v[inside])\\n    x = np.array([root(lambda a: (_calibration(a[None])@w-y).ravel(), y).x for y in x])\\n    pb = study[\'panel_b\']\\n    z = _design(x)@_ls(_design(pb[\'u\']), pb[\'v\'])\\n    pc = study[\'panel_c\']\\n    ops = []\\n    def odd(u, k):\\n        return -np.sign(u)*np.expm1(-k*abs(u))\\n    def even(u, k):\\n        u2 = u*u\\n        return u2/np.sqrt(1+k*u2)\\n    for transform, name, bounds in [(odd, \'v1\', (.5, 3)), (even, \'v2\', (.05, 1.5))]:\\n        u, v = pc[\'u\'], pc[name]\\n        def residual(rate):\\n            f = _design(u, transform(u, rate)) if name == \'v2\' else _design(transform(u, rate))\\n            return np.linalg.norm(v-f@_ls(f, v))\\n        k = minimize_scalar(residual, bounds=bounds, method=\'bounded\').x\\n        if name == \'v1\':\\n            ops.append(_design(transform(z, k))@_ls(_design(transform(u, k)), v))\\n        else:\\n            ops.append(_design(z, transform(z, k))@_ls(_design(u, transform(u, k)), v))\\n    return np.column_stack((z, *ops))\\n\\n\\ndef _kernel(x, y):\\n    i, j = np.triu_indices(x.shape[1])\\n    px, py = x[:, i]*x[:, j], y[:, i]*y[:, j]\\n    return (1 + x@y.T + .1*(px@py.T)\\n            + .4*np.exp(-cdist(x, y, metric=\'sqeuclidean\')/60))\\n\\n\\ndef _predict(v, n, y, state):\\n    # The second auxiliary output is useful, but its quadratic products\\n    # extrapolate less reliably than those of the primary and odd branches.\\n    weights = np.repeat([2., 2., .35], 4)\\n    z = v * (weights / np.maximum(v[:n].std(0), .001))\\n    train = z[:n]\\n    alpha = solve(_kernel(train, train) + .004*np.eye(n),\\n                  np.column_stack((y, state)), assume_a=\'pos\')\\n    pred = _kernel(z[n:], train) @ alpha\\n    return pred[:, 0], pred[:, 1:]\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    tr = study[\'train\']\\n    v = _coordinates(study, np.vstack((tr[\'x\'], study[\'test\'][\'x\'])))\\n    y, state = _predict(v, len(tr[\'x\']), tr[\'y\'], tr[\'traj\'])\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n"}'

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
