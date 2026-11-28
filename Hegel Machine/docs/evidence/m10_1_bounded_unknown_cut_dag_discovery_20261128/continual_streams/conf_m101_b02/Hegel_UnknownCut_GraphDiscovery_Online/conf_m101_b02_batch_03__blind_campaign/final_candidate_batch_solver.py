"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "2f57e21aea2b3d6957b94495a74ad507259b29cce319e4e2c62d519aeaadbb61"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrate the panel graph and fit sparse scalar and trajectory readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar, root\\nfrom sklearn.linear_model import ARDRegression\\n\\n\\ndef _design(*blocks):\\n    return np.column_stack((np.ones(len(blocks[0])), *blocks))\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    a = _design(u)\\n    keep = np.argsort(np.linalg.norm(a @ _ls(a, v) - v, axis=1))[:45]\\n    u, v = u[keep], v[keep]\\n    cu = np.roll(u, -1, axis=1)\\n    w = np.column_stack([_ls(_design(u[:, j], cu[:, j], u[:, j] * cu[:, j], u[:, j] ** 3), v[:, j]) for j in range(4)])\\n\\n    def forward(z):\\n        q = np.roll(z, -1)\\n        return w[0] + w[1] * z + w[2] * q + w[3] * z * q + w[4] * z ** 3\\n\\n    starts = _design(x) @ _ls(_design(v), u)\\n    return np.array([root(lambda z: forward(z) - row, start, tol=1e-11).x\\n                     for row, start in zip(x, starts)])\\n\\n\\ndef _odd(z, rate):\\n    return _design(-np.sign(z) * np.expm1(-rate * np.abs(z)))\\n\\n\\ndef _even(z, rate):\\n    s = z * z\\n    return _design(z, s / np.sqrt(1 + rate * s))\\n\\n\\ndef _operator(u, v, f, design, bounds):\\n    def loss(rate):\\n        g = design(u, rate)\\n        return np.linalg.norm(g @ _ls(g, v) - v)\\n\\n    rate = minimize_scalar(loss, bounds=bounds, method=\'bounded\').x\\n    return design(f, rate) @ _ls(design(u, rate), v)\\n\\n\\ndef _features(study, x):\\n    z = _calibrate(study[\'panel_a\'], x)\\n    b = study[\'panel_b\']\\n    f = _design(z) @ _ls(_design(b[\'u\']), b[\'v\'])\\n    p = study[\'panel_c\']\\n    c = _operator(p[\'u\'], p[\'v1\'], f, _odd, (.9, 2.7))\\n    d = _operator(p[\'u\'], p[\'v2\'], f, _even, (.1945, .5835))\\n    z = np.column_stack((f, c, d))\\n    i, j = np.triu_indices(z.shape[1])\\n    return np.column_stack((z, z[:, i] * z[:, j])), np.column_stack((np.sin(f), np.cos(f), np.sin(c), np.cos(c), np.sin(d), np.cos(d)))\\n\\n\\ndef _time_basis(times, traj):\\n    def design(rates):\\n        return -np.expm1(-times[:, None] * rates)\\n\\n    def residual(rates):\\n        h = design(rates)\\n        return (h @ _ls(h, traj.T) - traj.T).ravel()\\n\\n    return design(least_squares(residual, [1.8, .42], bounds=(.001, 15)).x)\\n\\n\\ndef _ard(a, b, y):\\n    mean, scale = a.mean(0), np.maximum(a.std(0), 1e-10)\\n    a, b = (a - mean) / scale, (b - mean) / scale\\n    return ARDRegression(threshold_lambda=10000).fit(a, y).predict(b)\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    g, extra = _features(study, np.vstack((train[\'x\'], study[\'test\'][\'x\'])))\\n    y = _ard(g[:n], g[n:], train[\'y\'])\\n    h = _time_basis(study[\'eval_times\'], train[\'traj\'])\\n    amplitudes = _ls(h, train[\'traj\'].T)\\n    wide = np.column_stack((g, extra))\\n    pred = np.column_stack([\\n        .75 * _ard(wide[:n], wide[n:], a) + .25 * _ard(g[:n], g[n:], a)\\n        for a in amplitudes\\n    ]) @ h.T\\n    terminal = pred[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': pred,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}\\n"}'

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
