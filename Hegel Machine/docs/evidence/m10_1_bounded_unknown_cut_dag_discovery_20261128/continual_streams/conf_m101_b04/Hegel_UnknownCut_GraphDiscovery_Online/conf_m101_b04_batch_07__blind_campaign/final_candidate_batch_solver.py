"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "4acb84973a96d454d56d2dc80815f4d85797698b859f00f0ba5e3f2dd29b6319", "node_01": "a7275a867ea6a5687b8b7b14c8fc80075628ff70545b78475355c4afe9b5241a"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares, root\\n\\n\\ndef _fit(panel):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v\'])\\n    neighbor = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, (u - 1.) * neighbor, u**3), axis=-1)\\n    weights = np.ones(len(u))\\n\\n    def fit(rows, w):\\n        return np.array([\\n            np.linalg.lstsq(design[rows, j] * w[:, None], v[rows, j] * w, rcond=None)[0]\\n            for j in range(u.shape[1])\\n        ]).T\\n\\n    rows = np.arange(len(u))\\n    for _ in range(20):\\n        coef = fit(rows, weights)\\n        error = np.linalg.norm(np.einsum(\'njk,kj->nj\', design, coef) - v, axis=1)\\n        scale = max(np.median(error), 1.e-4)\\n        weights = 1. / np.hypot(1., error / scale)\\n    # The calibration protocol introduces exactly three contaminated rows.\\n    keep = np.argsort(error)[:-3]\\n    return fit(keep, np.ones(len(keep)))\\n\\n\\ndef _invert(v, coef):\\n    bias, gain, cross, cubic = coef\\n    j = np.arange(len(bias))\\n    nxt = (j + 1) % len(j)\\n    linear = np.diag(gain)\\n    linear[j, nxt] = -cross\\n\\n    def forward(x):\\n        return bias + gain * x + cross * (x - 1.) * x[nxt] + cubic * x**3\\n\\n    def jac(x):\\n        jacobian = np.diag(gain + cross * x[nxt] + 3. * cubic * x**2)\\n        jacobian[j, nxt] = cross * (x - 1.)\\n        return jacobian\\n\\n    seed = np.linalg.solve(linear, (v - bias).T).T\\n    result = np.empty_like(seed)\\n    for i, target in enumerate(v):\\n        def residual(x):\\n            return forward(x) - target\\n\\n        start = np.clip(seed[i], -1.5, 1.5)\\n        sol = root(residual, start, jac=jac, tol=1.e-10)\\n        if np.linalg.norm(sol.fun, ord=np.inf) > 1.e-9:\\n            sol = least_squares(residual, start, jac=jac, gtol=1.e-12)\\n        result[i] = sol.x\\n    return result\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coef = _fit(study[\'panel_a\'])\\n    return {\'x_tr\': _invert(study[\'train\'][\'x\'], coef),\\n            \'x_te\': _invert(study[\'test\'][\'x\'], coef)}\\n", "node_01": "import numpy as np\\nfrom scipy.linalg import solve\\nfrom scipy.optimize import least_squares\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _panel_coordinates(panel, x):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v\'])\\n\\n    def design(a, scale):\\n        return np.column_stack((np.ones(len(a)), a, a**3 / (1.0 + scale * a**2)))\\n\\n    def fit(scale):\\n        d = design(u, scale)\\n        return d, np.linalg.lstsq(d, v, rcond=None)[0]\\n\\n    def residual(scale):\\n        d, coef = fit(scale)\\n        return (d @ coef - v).ravel()\\n\\n    scale = least_squares(residual, [0.75], bounds=(0.015, 10.0)).x\\n    _, coef = fit(scale)\\n    return design(x, scale) @ coef\\n\\n\\ndef _scalar(b, bt, y):\\n    n = len(b)\\n    coords = np.vstack((b, bt))\\n    trend = np.column_stack((np.ones(len(coords)), coords))\\n    train = trend[:n]\\n    # Use the fitted linear score as the argument of bounded link corrections.\\n    beta = np.linalg.lstsq(train, y, rcond=None)[0]\\n    h = (trend @ beta - y.mean()) / y.std()\\n    links = (\\n        np.column_stack((np.tanh(h), np.exp(-h*h), h*h/(1+h*h))),\\n        np.column_stack((np.sin(h), np.cos(h), np.sin(2*h), np.cos(2*h))),\\n    )\\n    pair = (coords[:, 1] * coords[:, 3])[:, None]\\n    quadratic = np.column_stack(\\n        [coords[:, i] * coords[:, j] for i in range(4) for j in range(i+1, 4)]\\n        + [coords**2]\\n    )\\n    scaled = coords / (np.sqrt(b.shape[1]) * b.std(axis=0))\\n    base = np.exp(-0.5 * cdist(scaled, scaled[:n], \'sqeuclidean\'))\\n    rhs = np.column_stack((y, train))\\n    predictions = []\\n    for link in links:\\n        for interaction in (pair, quadratic):\\n            features = np.column_stack((link, interaction))\\n            features /= np.maximum(features[:n].std(axis=0), 0.01)\\n            kernel = base + 0.1 * features @ features[:n].T\\n            for penalty in (0.01, 0.03, 0.08):\\n                p = solve(kernel[:n] + penalty * np.eye(n), rhs, assume_a=\'pos\')\\n                beta = np.linalg.solve(train.T @ p[:, 1:], train.T @ p[:, 0])\\n                predictions.append(trend[n:] @ beta + kernel[n:] @ (p[:, 0] - p[:, 1:] @ beta))\\n    return np.mean(predictions, axis=0)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study, calibrated = inputs[\'__input__\'], inputs[\'node_00\']\\n    x, xt = calibrated[\'x_tr\'], calibrated[\'x_te\']\\n    b, bt = np.split(_panel_coordinates(study[\'panel_b\'], np.vstack((x, xt))), [len(x)])\\n    y_hat = _scalar(b, bt, np.asarray(study[\'train\'][\'y\']))\\n    d = np.column_stack((np.ones(len(x)), x, b))\\n    dt = np.column_stack((np.ones(len(xt)), xt, bt))\\n    state_pred = dt @ np.linalg.lstsq(d, study[\'train\'][\'traj\'], rcond=None)[0]\\n    end = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
