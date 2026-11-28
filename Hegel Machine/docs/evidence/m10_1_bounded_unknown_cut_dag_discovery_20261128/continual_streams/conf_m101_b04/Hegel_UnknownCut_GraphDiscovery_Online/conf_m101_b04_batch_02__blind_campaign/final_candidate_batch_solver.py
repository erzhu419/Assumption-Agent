"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_00", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "1771cfb1d93dfe6c18fc7270381edec0a047d5ac6b667d666ccb1b27f7943ca7"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Fit separate nonlinear scalar and trajectory readouts from the study.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve, qr\\nfrom scipy.optimize import least_squares, minimize\\n\\n\\ndef _monomials(x, degree):\\n    return np.column_stack([np.prod(x[:, cols], axis=1) for cols in\\n                            combinations_with_replacement(range(x.shape[1]), degree)])\\n\\n\\ndef _quadratic(x):\\n    return np.column_stack((np.ones(len(x)), x, _monomials(x, 2)))\\n\\n\\ndef _gaussian(x, z):\\n    return np.exp(-0.23 * np.sum((x[:, None] - z[None, :]) ** 2, axis=2))\\n\\n\\ndef _predict(p, pt, k, kt, target, penalty):\\n    f = cho_factor(k + penalty * np.eye(len(k)))\\n    a = cho_solve(f, p)\\n    v = cho_solve(f, target)\\n    trend = np.linalg.solve(p.T @ a, p.T @ v)\\n    return pt @ trend + kt @ (v - a @ trend)\\n\\n\\ndef _matern_readout(x, xt, p, pt, y):\\n    # Profile out the quadratic mean before fitting the input length scales.\\n    n = qr(p, mode=\'full\')[0][:, p.shape[1]:]\\n    v = n.T @ y\\n    delta = (x[:, None] - x[None, :]) ** 2\\n    delta_test = (xt[:, None] - x[None, :]) ** 2\\n\\n    def kernel(distance_sq, par):\\n        r2 = distance_sq @ np.exp(par[:x.shape[1]])\\n        r = np.sqrt(r2)\\n        return (1 + r + 0.4 * r2 + r2 * r / 15) * np.exp(-r)\\n\\n    def objective(par):\\n        k = n.T @ kernel(delta, par) @ n + np.exp(par[-1]) * np.eye(len(v))\\n        f = cho_factor(k)\\n        return np.log(v @ cho_solve(f, v) / len(v)) + 2 * np.log(np.diag(f[0])).mean()\\n\\n    par = minimize(objective, np.log([0.24] * x.shape[1] + [0.002]),\\n                   method=\'L-BFGS-B\', bounds=[(-4, 1)] * x.shape[1] + [(-12, -1)],\\n                   options={\'maxiter\': 70}).x\\n    return _predict(p, pt, kernel(delta, par), kernel(delta_test, par), y, np.exp(par[-1]))\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    x, xt = train[\'x\'], study[\'test\'][\'x\']\\n    t = study[\'eval_times\']\\n    trajectories = train[\'traj\'].T\\n\\n    def basis(rates):\\n        return -np.expm1(-t[:, None] * rates)\\n\\n    def residual(rates):\\n        b = basis(rates)\\n        amplitudes = np.linalg.lstsq(b, trajectories, rcond=None)[0]\\n        return (b @ amplitudes - trajectories).ravel()\\n\\n    rates = least_squares(residual, [1.5, 0.25],\\n                          bounds=([0.7, 0.02], [4.0, 0.65])).x\\n    modes = np.linalg.qr(basis(rates))[0]\\n    p, pt = _quadratic(x), _quadratic(xt)\\n    k, kt = _gaussian(x, x), _gaussian(xt, x)\\n    y = train[\'y\']\\n    y_hat = 0.5 * (_predict(p, pt, k, kt, y, 0.001)\\n                   + _matern_readout(x, xt, p, pt, y))\\n    c, ct = _monomials(x, 3), _monomials(xt, 3)\\n    amplitudes = _predict(p, pt, k + 0.05 * (c @ c.T),\\n                          kt + 0.05 * (ct @ c.T), trajectories.T @ modes, 0.006)\\n    state_pred = amplitudes @ modes.T\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(),\\n                                         terminal.mean(), terminal.std()])}\\n"}'

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
