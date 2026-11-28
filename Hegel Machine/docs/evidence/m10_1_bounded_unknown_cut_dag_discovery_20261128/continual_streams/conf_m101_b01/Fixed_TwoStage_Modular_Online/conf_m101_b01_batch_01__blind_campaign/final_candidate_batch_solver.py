"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "8176e87eb3ce1eedc663eb589a42af7dbe673aebde9f19052a39b7225bf7015e", "node_01": "769209d476f51d630145f9b63ccca6cf1d070a465897d742cceca413b2f7221d"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(u):\\n    return np.column_stack((np.ones(len(u)), u))\\n\\n\\ndef _panel_a(panel):\\n    d, v = _affine(panel[\'u\']), panel[\'v\']\\n    w = np.linalg.lstsq(d, v, rcond=None)[0]\\n    for _ in range(20):\\n        r = np.linalg.norm(d @ w - v, axis=1)\\n        scale = max(np.median(r), 1e-9)\\n        q = 1 / np.hypot(1, r / scale)\\n        w = np.linalg.lstsq(d * q[:, None], v * q[:, None], rcond=None)[0]\\n    return w\\n\\n\\ndef _panel_b(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def basis(x, params):\\n        amplitude, frequency = params\\n        return _affine(x - amplitude * np.sin(frequency * x))\\n\\n    def residual(params):\\n        d = basis(u, params)\\n        return (d @ np.linalg.lstsq(d, v, rcond=None)[0] - v).ravel()\\n\\n    params = least_squares(residual, [0.4, 2.87],\\n                           bounds=([0.0, 1.5], [0.8, 4.5])).x\\n    w = np.linalg.lstsq(basis(u, params), v, rcond=None)[0]\\n    return lambda x: basis(x, params) @ w\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a = _panel_a(study[\'panel_a\'])\\n    b = _panel_b(study[\'panel_b\'])\\n\\n    def coordinates(x):\\n        return b(np.linalg.solve(a[1:].T, (x - a[0]).T).T)\\n\\n    train, test = study[\'train\'], study[\'test\']\\n    return {\'z_tr\': coordinates(train[\'x\']), \'z_te\': coordinates(test[\'x\']),\\n            \'train_y\': train[\'y\'], \'train_traj\': train[\'traj\'],\\n            \'eval_times\': study[\'eval_times\']}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\nfrom scipy.special import hyp2f1\\n\\n\\ndef _basis(times, rates):\\n    return -np.expm1(-np.exp(rates)[:, None] * times)\\n\\n\\ndef _time_fit(times, traj):\\n    def residual(rates):\\n        b = _basis(times, rates)\\n        return (traj @ np.linalg.pinv(b) @ b - traj).ravel()\\n    fit = least_squares(residual, [-1.0, 1.0], bounds=(-8.0, 4.0))\\n    return _basis(times, np.sort(fit.x))\\n\\n\\ndef _features(x, p):\\n    a = x * np.exp(p[0]) + p[1:5]\\n    h = a * hyp2f1(0.5, p[5], 1.5, -a*a)\\n    state = np.column_stack((np.ones(len(x)), h, h*h))\\n    left, right = np.triu_indices(4, 1)\\n    readout = np.column_stack((state, h[:, left] * h[:, right]))\\n    return readout, state\\n\\n\\ndef _spatial_fit(x, targets):\\n    scale = np.maximum(0.05, targets.std(axis=0))\\n    normalized = targets / scale\\n    def profile(d, v):\\n        penalty = np.diag(np.r_[0., np.full(d.shape[1] - 1, 0.02)])\\n        w = np.linalg.lstsq(np.vstack((d, penalty)),\\n                            np.vstack((v, np.zeros((len(penalty), v.shape[1])))),\\n                            rcond=None)[0]\\n        return w, (d @ w - v).ravel(), (penalty @ w).ravel()\\n\\n    def residual(p):\\n        d, g = _features(x, p)\\n        _, e1, r1 = profile(d, normalized[:, :1])\\n        _, e2, r2 = profile(g, normalized[:, 1:])\\n        return np.r_[e1, e2, r1, r2, 0.07*p[1:5]]\\n\\n    fit = least_squares(residual, [np.log(3.), 0., 0., 0., 0., 0.5],\\n                        bounds=([-1.5] + [-0.45]*4 + [0.2], [1.9] + [0.45]*4 + [1.3]),\\n                        max_nfev=75)\\n    d, g = _features(x, fit.x)\\n    w, _, _ = profile(d, normalized[:, :1])\\n    v, _, _ = profile(g, normalized[:, 1:])\\n    return fit.x, w[:, 0] * scale[0], v * scale[1:]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    traj = data[\'train_traj\']\\n    b = _time_fit(data[\'eval_times\'], traj)\\n    targets = np.column_stack((data[\'train_y\'], traj @ np.linalg.pinv(b)))\\n    p, w, v = _spatial_fit(data[\'z_tr\'], targets)\\n    d, g = _features(data[\'z_te\'], p)\\n    y, state = d @ w, g @ v @ b\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}\\n"}'

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
