"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "34991e7e985efcfb3261a4c4bd5d78a5c7b340e03f32d3374a653f709cada055"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _linear(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _kernel_fit(x, xt, trend, trendt, y, length, nugget):\\n    factor = 2 * x.shape[1] * length**2\\n    k = np.exp(-cdist(x, x, \'sqeuclidean\') / factor)\\n    k.flat[::len(x) + 1] += nugget\\n    solved = cho_solve(cho_factor(k), np.column_stack((trend, y)))\\n    w, v = solved[:, :-1], solved[:, -1]\\n    beta = np.linalg.solve(trend.T @ w, trend.T @ v)\\n    cross = np.exp(-cdist(xt, x, \'sqeuclidean\') / factor)\\n    return trendt @ beta + cross @ (v - w @ beta)\\n\\n\\ndef _features(x, xt, y, trajectories):\\n    scale = x.std(axis=0)\\n    x, xt = x / scale, xt / scale\\n    d, dt = _linear(x), _linear(xt)\\n    _, _, vt = np.linalg.svd(trajectories, full_matrices=False)\\n    profile = vt[0]\\n    z = trajectories @ profile\\n    zp = _kernel_fit(x, xt, d, dt, z, 1.7, 1e-4)\\n    pairs = list(combinations_with_replacement(range(x.shape[1]), 2))\\n    quadratic = np.column_stack([d] + [x[:, i] * x[:, j] for i, j in pairs])\\n    coef = np.linalg.lstsq(quadratic, y, rcond=None)[0][d.shape[1]:]\\n    h = np.zeros((x.shape[1], x.shape[1]))\\n    for (i, j), c in zip(pairs, coef):\\n        h[i, j] = h[j, i] = c / (1 + (i != j))\\n    val, vec = np.linalg.eigh(h)\\n    importance = abs(val) / max(abs(val))\\n    return x, xt, z / z.std(), zp / z.std(), importance, vec, zp[:, None] * profile\\n\\n\\ndef _predict(features, y, cfg):\\n    x, xt, z, zp, imp, vec, _ = features\\n    mix, stretch, warp, latent_metric, use_latent, length, nugget = cfg\\n    d, dt = _linear(x), _linear(xt)\\n    if use_latent:\\n        d, dt = np.column_stack((d, z)), np.column_stack((dt, zp))\\n    if warp:\\n        x, xt = x + warp * x*x, xt + warp * xt*xt\\n    if mix:\\n        weights = np.sqrt((1 - mix) + mix * (0.6 + stretch * imp))\\n        x, xt = (x @ vec) * weights, (xt @ vec) * weights\\n    if latent_metric:\\n        # Keep the original kernel\'s distance normalization after adding a coordinate.\\n        scale = np.sqrt((x.shape[1] + 1) / x.shape[1])\\n        x = np.column_stack((x, latent_metric * z)) * scale\\n        xt = np.column_stack((xt, latent_metric * zp)) * scale\\n    return _kernel_fit(x, xt, d, dt, y, length, nugget)\\n\\n\\ndef solve_batch(study):\\n    train = study[\'train\']\\n    y = train[\'y\']\\n    features = _features(train[\'x\'], study[\'test\'][\'x\'], y, train[\'traj\'])\\n    state = features[-1]\\n    pred = sum(w * _predict(features, y, cfg) for w, cfg in _SCALAR_MODELS)\\n    last = state[:, -1]\\n    return {\'y_hat\': pred, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([pred.mean(), pred.std(), last.mean(), last.std()])}\\n\\n\\n_SCALAR_MODELS = [(0.375, (1.0, 0.4, 0.0, 0.55, 0, 1.12, 3e-06)), (0.375, (1.0, 1.5, -0.075, 0.0, 1, 0.82, 3e-06)), (0.25, (1.0, 0.4, 0.075, 0.55, 0, 1.34, 3e-06))]\\n"}'

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
