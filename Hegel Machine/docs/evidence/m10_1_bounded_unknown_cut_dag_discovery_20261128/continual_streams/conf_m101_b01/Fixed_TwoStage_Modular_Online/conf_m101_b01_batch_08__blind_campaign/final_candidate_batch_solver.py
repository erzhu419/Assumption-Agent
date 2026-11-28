"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "a9ec3ca6e864c1f0b81ca7ae124e505da93347ffede520e2090b1572a1551ef1"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover calibrated coordinates and predict the two public observables.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _next(x):\\n    return np.roll(x, -1, axis=1)\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    keep = np.argsort(np.linalg.norm(v - _affine(u) @ _fit(_affine(u), v), axis=1))[:45]\\n\\n    def design(x, j, frequency=None):\\n        p, q = x[:, j], x[:, (j + 1) % 4]\\n        base = np.column_stack((np.ones(len(x)), p, q, p*q))\\n        if frequency is None:\\n            return base\\n        return np.column_stack((base, np.sin(frequency*p), np.sin(2*frequency*p)))\\n\\n    for _ in range(5):\\n        residual = np.column_stack([\\n            design(u, j) @ _fit(design(u[keep], j), v[keep, j]) - v[:, j]\\n            for j in range(4)\\n        ])\\n        keep = np.argsort(np.linalg.norm(residual, axis=1))[:45]\\n\\n    def profile(frequency):\\n        weights, errors = [], []\\n        for j in range(4):\\n            matrix = design(u[keep], j, frequency)\\n            pinv = np.linalg.pinv(matrix)\\n            w = _fit(matrix, v[keep, j])\\n            leverage = np.sum(matrix * pinv.T, axis=1)\\n            errors.append(np.mean(abs(v[keep, j] - matrix @ w)/(1-leverage)))\\n            weights.append(w)\\n        return np.mean(errors), weights\\n\\n    grid = np.linspace(.35, 8, 32)\\n    j = np.argmin([profile(f)[0] for f in grid])\\n    frequency = minimize_scalar(lambda f: profile(f)[0], method=\'bounded\',\\n                                bounds=(grid[max(0, j-1)], grid[min(len(grid)-1, j+1)])).x\\n    weights = profile(frequency)[1]\\n\\n    def forward(x):\\n        return np.column_stack([design(x, j, frequency) @ weights[j] for j in range(4)])\\n\\n    def inverse(rows):\\n        return np.array([least_squares(lambda z: forward(z[None])[0] - row,\\n                                       row, max_nfev=70).x for row in rows])\\n    return inverse\\n\\n\\ndef _coordinates(study, x):\\n    z = _calibration(study[\'panel_a\'])(x)\\n    panel = study[\'panel_b\']\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def warp(x, p):\\n        f, a, g, h = p\\n        return _affine(x + a*np.sin(f*x) + g*np.tanh(h*x))\\n\\n    def residual(p):\\n        matrix = warp(u, p)\\n        return (matrix @ _fit(matrix, v) - v).ravel()\\n\\n    p = least_squares(residual, [3., -.3, -.05, 2.], max_nfev=150).x\\n    b = warp(z, p) @ _fit(warp(u, p), v)\\n    panel = study[\'panel_c\']\\n    u = panel[\'u\']\\n    c = _affine(b*_next(b)) @ _fit(_affine(u*_next(u)), panel[\'v1\'])\\n    return z, c\\n\\n\\ndef _readout(z, c, y, traj):\\n    n = len(y)\\n    # The scalar and trajectory share a cyclic interaction with tied weights.\\n    cyclic = (z * np.roll(c, 1, axis=1)).sum(axis=1)\\n    state_design = _affine(np.column_stack((c, cyclic)))\\n    scalar_design = np.column_stack((state_design, c[:, 0] * c[:, 2]))\\n    yhat = scalar_design[n:] @ _fit(scalar_design[:n], y)\\n    state = state_design[n:] @ _fit(state_design[:n], traj)\\n    return yhat, state\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    z, c = _coordinates(study, np.concatenate((train[\'x\'], study[\'test\'][\'x\'])))\\n    y, state = _readout(z, c, train[\'y\'], train[\'traj\'])\\n    terminal = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}\\n"}'

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
