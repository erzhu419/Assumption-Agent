"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "4acb84973a96d454d56d2dc80815f4d85797698b859f00f0ba5e3f2dd29b6319", "node_01": "97417aa4137ebd55cdd9a777f4bd5e0659e808e7ca528cd61ab67b175e987e02"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares, root\\n\\n\\ndef _fit(panel):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v\'])\\n    neighbor = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, (u - 1.) * neighbor, u**3), axis=-1)\\n    weights = np.ones(len(u))\\n\\n    def fit(rows, w):\\n        return np.array([\\n            np.linalg.lstsq(design[rows, j] * w[:, None], v[rows, j] * w, rcond=None)[0]\\n            for j in range(u.shape[1])\\n        ]).T\\n\\n    rows = np.arange(len(u))\\n    for _ in range(20):\\n        coef = fit(rows, weights)\\n        error = np.linalg.norm(np.einsum(\'njk,kj->nj\', design, coef) - v, axis=1)\\n        scale = max(np.median(error), 1.e-4)\\n        weights = 1. / np.hypot(1., error / scale)\\n    # The calibration protocol introduces exactly three contaminated rows.\\n    keep = np.argsort(error)[:-3]\\n    return fit(keep, np.ones(len(keep)))\\n\\n\\ndef _invert(v, coef):\\n    bias, gain, cross, cubic = coef\\n    j = np.arange(len(bias))\\n    nxt = (j + 1) % len(j)\\n    linear = np.diag(gain)\\n    linear[j, nxt] = -cross\\n\\n    def forward(x):\\n        return bias + gain * x + cross * (x - 1.) * x[nxt] + cubic * x**3\\n\\n    def jac(x):\\n        jacobian = np.diag(gain + cross * x[nxt] + 3. * cubic * x**2)\\n        jacobian[j, nxt] = cross * (x - 1.)\\n        return jacobian\\n\\n    seed = np.linalg.solve(linear, (v - bias).T).T\\n    result = np.empty_like(seed)\\n    for i, target in enumerate(v):\\n        def residual(x):\\n            return forward(x) - target\\n\\n        start = np.clip(seed[i], -1.5, 1.5)\\n        sol = root(residual, start, jac=jac, tol=1.e-10)\\n        if np.linalg.norm(sol.fun, ord=np.inf) > 1.e-9:\\n            sol = least_squares(residual, start, jac=jac, gtol=1.e-12)\\n        result[i] = sol.x\\n    return result\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coef = _fit(study[\'panel_a\'])\\n    return {\'x_tr\': _invert(study[\'train\'][\'x\'], coef),\\n            \'x_te\': _invert(study[\'test\'][\'x\'], coef)}\\n", "node_01": "from itertools import combinations\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _operator(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    candidates = []\\n    for fun, start, limits in ((np.sin, 1.6, (.25, 3.5)),\\n                               (np.tanh, 2.5, (.25, 6.0))):\\n        def design(x, rate):\\n            return np.column_stack((_affine(x), fun(x * rate)))\\n\\n        def residual(rate):\\n            a = design(u, rate)\\n            return (a @ np.linalg.lstsq(a, v, rcond=None)[0] - v).ravel()\\n\\n        rate = least_squares(residual, [start], bounds=limits).x[0]\\n        a = design(u, rate)\\n        coef = np.linalg.lstsq(a, v, rcond=None)[0]\\n        q = np.linalg.qr(a)[0]\\n        loo = (v - a @ coef) / (1.0 - np.sum(q*q, axis=1))[:, None]\\n        candidates.append((np.mean(loo**2), fun, rate, coef))\\n    _, fun, rate, coef = min(candidates, key=lambda item: item[0])\\n    return lambda x: np.column_stack((_affine(x), fun(x * rate))) @ coef\\n\\n\\ndef _ridge(a, y, penalty):\\n    extra = np.diag(np.r_[0., np.full(a.shape[1] - 1, np.sqrt(penalty))])\\n    return np.linalg.lstsq(np.vstack((a, extra)),\\n                           np.r_[y, np.zeros((len(extra),) + y.shape[1:])],\\n                           rcond=None)[0]\\n\\n\\ndef _scalar_readout(z, zt, y):\\n    pairs = list(combinations(range(z.shape[1]), 2))\\n    products = np.column_stack([z[:, i] * z[:, j] for i, j in pairs])\\n    linear = _affine(z)\\n\\n    def loss(cols):\\n        design = np.column_stack((linear, products[:, cols]))\\n        residual = design @ _ridge(design, y, .001) - y\\n        return np.mean(residual**2)\\n\\n    cols = min(combinations(range(len(pairs)), 2), key=loss)\\n    pt = np.column_stack([zt[:, pairs[k][0]] * zt[:, pairs[k][1]] for k in cols])\\n    p = products[:, cols]\\n    design = np.column_stack((linear, p, p**2))\\n    test_design = np.column_stack((_affine(zt), pt, pt**2))\\n    return test_design @ _ridge(design, y, .015)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    calibrated = inputs[\'node_00\']\\n    train = study[\'train\']\\n    panel_b = _operator(study[\'panel_b\'])\\n    z, zt = [panel_b(x) for x in (calibrated[\'x_tr\'], calibrated[\'x_te\'])]\\n    y_hat = _scalar_readout(z, zt, train[\'y\'])\\n\\n    times, observed = study[\'eval_times\'], train[\'traj\'].T\\n\\n    def basis(rates):\\n        return -np.expm1(-times[:, None] * rates)\\n\\n    def residual(rates):\\n        b = basis(rates)\\n        return (b @ np.linalg.lstsq(b, observed, rcond=None)[0] - observed).ravel()\\n\\n    rates = least_squares(residual, [1.8, .4], bounds=([.7, .03], [5., .68])).x\\n    modes = np.linalg.qr(basis(rates))[0]\\n    state_pred = _affine(zt) @ _ridge(_affine(z), observed.T @ modes, .001) @ modes.T\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}\\n"}'

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
