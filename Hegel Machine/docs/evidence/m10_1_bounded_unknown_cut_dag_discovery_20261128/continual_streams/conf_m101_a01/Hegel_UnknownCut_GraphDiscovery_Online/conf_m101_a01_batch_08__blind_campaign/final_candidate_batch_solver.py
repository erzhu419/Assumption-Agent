"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "e5783a6059695aaab685c8a12bf8567e900b7ffce431f66457ebd43a95b6efd9", "node_01": "af0a9e7cc9d34c1645ea8a9d0edffa696216ded8dbd22404a6ad4c67e2f05c66"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _linear(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x, xt = study[\'train\'][\'x\'], study[\'test\'][\'x\']\\n    a = study[\'panel_a\']\\n    scale, bias = [], []\\n    for u, v in zip(a[\'u\'].T, a[\'v\'].T):\\n        i, j = np.triu_indices(len(u), 1)\\n        dx = u[i] - u[j]\\n        keep = np.abs(dx) > 1e-9\\n        slope = np.median((v[i] - v[j])[keep] / dx[keep])\\n        scale.append(slope)\\n        bias.append(np.median(v - slope * u))\\n    b = study[\'panel_b\']\\n    coef = np.linalg.lstsq(_linear(b[\'u\']), b[\'v\'], rcond=None)[0]\\n    q = _linear((x - bias) / scale) @ coef\\n    qt = _linear((xt - bias) / scale) @ coef\\n    return {\'q_tr\': q, \'q_te\': qt}\\n", "node_01": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _quadratic(z):\\n    return np.column_stack((_affine(z), *[z[:, i] * z[:, j] for i, j in combinations_with_replacement(range(z.shape[1]), 2)]))\\n\\n\\ndef _readout(u, v, z):\\n    def design(a, w):\\n        return np.column_stack((_affine(a), np.cos(w * a)))\\n\\n    def fit(w):\\n        a = design(u, w)\\n        return a, np.linalg.lstsq(a, v, rcond=None)[0]\\n\\n    def loss(w):\\n        a, coef = fit(w)\\n        return np.mean((a @ coef - v) ** 2)\\n\\n    w = minimize_scalar(loss, bounds=(0.6, 2.2), method=\'bounded\').x\\n    _, coef = fit(w)\\n    return design(z, w) @ coef\\n\\n\\ndef _kernel(a, b):\\n    d = ((a[:, None] - b[None, :]) ** 2).sum(axis=2)\\n    return np.exp(-0.5 * d)\\n\\n\\ndef _response_trend(train_basis, test_basis, y):\\n    # Profile out the affine trend and the amplitude of the saturating term.\\n    affine = train_basis[:, :5]\\n    terms = train_basis[:, 1:]\\n    orth = np.linalg.qr(affine)[0]\\n    target = y - orth @ (orth.T @ y)\\n\\n    def loss(p):\\n        h = np.tanh(p[0] + terms @ p[1:])\\n        h -= orth @ (orth.T @ h)\\n        amplitude = (h @ target) / (h @ h + 1e-10)\\n        return np.r_[amplitude * h - target, 0.002 * p]\\n\\n    direction = np.linalg.lstsq(train_basis, y, rcond=None)[0]\\n    rng = np.random.default_rng(15)\\n    starts = [direction, *rng.normal(0, 0.8, (7, len(direction)))]\\n    p = min((least_squares(loss, g, max_nfev=90) for g in starts),\\n            key=lambda fit: fit.cost).x\\n\\n    def design(basis):\\n        return np.column_stack((basis[:, :5], np.tanh(p[0] + basis[:, 1:] @ p[1:])))\\n\\n    a = design(train_basis)\\n    coef = np.linalg.lstsq(a, y, rcond=None)[0]\\n    return y - a @ coef, design(test_basis) @ coef\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study, calibrated = inputs[\'__input__\'], inputs[\'node_00\']\\n    q, qt = calibrated[\'q_tr\'], calibrated[\'q_te\']\\n    train, panel = study[\'train\'], study[\'panel_c\']\\n    basis, test_basis = _quadratic(q), _quadratic(qt)\\n    residual, trend = _response_trend(basis, test_basis, train[\'y\'])\\n    z = np.vstack((q, qt))\\n    z = np.column_stack((z, 0.2 * _readout(panel[\'u\'], panel[\'v2\'], z)))\\n    n = len(q)\\n    k = _kernel(z, z[:n])\\n    y_hat = trend + k[n:] @ np.linalg.solve(k[:n] + 0.003 * np.eye(n), residual)\\n    state_pred = test_basis @ np.linalg.lstsq(basis, train[\'traj\'], rcond=None)[0]\\n    end = state_pred[:, -1]\\n    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics_vec}\\n"}'

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
