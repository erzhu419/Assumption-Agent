"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "ca73c2c2ed69461751b0dba03683df4e9ca009197a700baf91152db4f48230da"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Predict the response and trajectory with separate kernel models.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _linear(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _rbf(x, xt, length):\\n    return np.exp(-cdist(x, xt, \'sqeuclidean\') / (2 * length**2))\\n\\n\\ndef _predict(k, cross, h, ht, target, nugget):\\n    k = k.copy()\\n    k.flat[::len(k) + 1] += nugget\\n    n = h.shape[1]\\n    w = cho_solve(cho_factor(k), np.column_stack((h, target)))\\n    a, b = w[:, :n], w[:, n:]\\n    beta = np.linalg.solve(h.T @ a, h.T @ b)\\n    return ht @ beta + cross @ (b - a @ beta)\\n\\n\\ndef _response(x, xt, y, h, ht):\\n    scale = x.std(axis=0)\\n    q, qt = x / scale, xt / scale\\n    return _predict(_rbf(q, q, 2), _rbf(qt, q, 2), h, ht, y[:, None], .001)[:, 0]\\n\\n\\ndef _trajectory(study, x, xt, z, h, ht):\\n    a = study[\'panel_a\']\\n    # Coordinatewise median slopes/intercepts reject sparse calibration outliers.\\n    scale, bias = [], []\\n    for u, v in zip(a[\'u\'].T, a[\'v\'].T):\\n        i, j = np.triu_indices(len(u), 1)\\n        dx = u[i] - u[j]\\n        keep = np.abs(dx) > 1e-9\\n        slope = np.median((v[i] - v[j])[keep] / dx[keep])\\n        scale.append(slope)\\n        bias.append(np.median(v - slope * u))\\n    b = study[\'panel_b\']\\n    coef = np.linalg.lstsq(_linear(b[\'u\']), b[\'v\'], rcond=None)[0]\\n    q = _linear((x - bias) / scale) @ coef\\n    qt = _linear((xt - bias) / scale) @ coef\\n    def features(u):\\n        return np.column_stack((u, np.tanh(.7 * u)))\\n    f, ft = features(q), features(qt)\\n    mu, sd = f.mean(axis=0), f.std(axis=0)\\n    f, ft = (f - mu) / sd, (ft - mu) / sd\\n    def kernel(p, r, u, v):\\n        return .25 * (1 + p @ r.T / f.shape[1])**2 + .2 * _rbf(u, v, 1.35)\\n    basis = np.linalg.svd(z, full_matrices=False)[2][:2]\\n    amplitudes = _predict(kernel(f, f, x, x), kernel(ft, f, xt, x), h, ht, z @ basis.T, .001)\\n    return amplitudes @ basis\\n\\n\\ndef solve_batch(study):\\n    train = study[\'train\']\\n    x, xt = train[\'x\'], study[\'test\'][\'x\']\\n    h, ht = _linear(x), _linear(xt)\\n    y = _response(x, xt, train[\'y\'], h, ht)\\n    state = _trajectory(study, x, xt, train[\'traj\'], h, ht)\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n"}'

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
