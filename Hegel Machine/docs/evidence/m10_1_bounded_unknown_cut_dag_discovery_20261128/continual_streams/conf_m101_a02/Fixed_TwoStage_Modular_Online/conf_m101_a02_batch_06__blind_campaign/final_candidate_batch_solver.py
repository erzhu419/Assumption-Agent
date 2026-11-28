"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "4c6c369c2c4853e1aebeaf38a2199f7ab481748d42a2efd10f2b387c09863dce", "node_01": "bc4191265402a17f209d8484b30a022f05c14b0760ac8b1db603aaad165a97de"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover reference coordinates and apply the second panel\'s nonlinear gain.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _a_design(x, j):\\n    a, b = x[:, j], x[:, (j + 1) % 4]\\n    return np.column_stack((np.ones(len(x)), a, b, a*b, a**3))\\n\\n\\ndef _fit_a(u, v):\\n    designs = [_a_design(u, j) for j in range(4)]\\n    weights = np.ones(len(u))\\n    for _ in range(20):\\n        coef = np.array([np.linalg.lstsq(d*weights[:, None], v[:, j]*weights, rcond=None)[0]\\n                         for j, d in enumerate(designs)])\\n        error = np.linalg.norm(np.column_stack([d@c for d, c in zip(designs, coef)]) - v, axis=1)\\n        weights = 1/np.hypot(1, error/max(np.median(error), 1e-8))\\n    keep = np.argsort(error)[:-3]\\n    coef = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]\\n                     for j, d in enumerate(designs)]).T\\n    inverse_seed = np.linalg.lstsq(_affine(v[keep]), u[keep], rcond=None)[0]\\n    return coef, inverse_seed\\n\\n\\ndef _invert_a(v, coef, inverse_seed):\\n    bias, lin, neighbor, pair, cubic = coef\\n    x = _affine(v) @ inverse_seed\\n    j = np.arange(4)\\n    for _ in range(20):\\n        b = np.roll(x, -1, axis=1)\\n        residual = bias + lin*x + neighbor*b + pair*x*b + cubic*x**3 - v\\n        if np.max(np.abs(residual), initial=0) < 1e-11:\\n            break\\n        jac = np.zeros((len(x), 4, 4))\\n        jac[:, j, j] = lin + pair*b + 3*cubic*x*x\\n        jac[:, j, (j+1)%4] = neighbor + pair*x\\n        x -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -0.5, 0.5)\\n    return x\\n\\n\\ndef _b_design(x, k):\\n    return np.column_stack((_affine(x), x*(-np.expm1(-k*x*x))))\\n\\n\\ndef _fit_b(u, v):\\n    def score(k):\\n        q = np.linalg.qr(_b_design(u, k))[0]\\n        err = (v - q @ (q.T @ v))/(1 - (q*q).sum(axis=1, keepdims=True))\\n        return np.mean(err**2)\\n    k = minimize_scalar(score, bounds=(0.15, 2.5), method=\'bounded\').x\\n    return k, np.linalg.lstsq(_b_design(u, k), v, rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    coef, seed = _fit_a(a[\'u\'], a[\'v\'])\\n    k, gain = _fit_b(b[\'u\'], b[\'v\'])\\n    return {out: _b_design(_invert_a(study[src][\'x\'], coef, seed), k) @ gain\\n            for out, src in ((\'z_tr\', \'train\'), (\'z_te\', \'test\'))}\\n", "node_01": "\\"\\"\\"Fit a shared nonlinear response for the endpoint and trajectory.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\nimport numpy as np\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _a_design(x, j):\\n    a, b = x[:, j], x[:, (j + 1) % 4]\\n    return np.column_stack((np.ones(len(x)), a, b, a*b, a**3))\\n\\n\\ndef _fit_a(u, v):\\n    designs = [_a_design(u, j) for j in range(4)]\\n    weights = np.ones(len(u))\\n    for _ in range(20):\\n        coef = np.array([np.linalg.lstsq(d*weights[:, None], v[:, j]*weights, rcond=None)[0]\\n                         for j, d in enumerate(designs)])\\n        error = np.linalg.norm(np.column_stack([d@c for d, c in zip(designs, coef)]) - v, axis=1)\\n        weights = 1/np.hypot(1, error/max(np.median(error), 1e-8))\\n    keep = np.argsort(error)[:-3]\\n    coef = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]\\n                     for j, d in enumerate(designs)]).T\\n    inverse_seed = np.linalg.lstsq(_affine(v[keep]), u[keep], rcond=None)[0]\\n    return coef, inverse_seed\\n\\n\\ndef _invert_a(v, coef, inverse_seed):\\n    bias, lin, neighbor, pair, cubic = coef\\n    x = _affine(v) @ inverse_seed\\n    j = np.arange(4)\\n    for _ in range(20):\\n        b = np.roll(x, -1, axis=1)\\n        residual = bias + lin*x + neighbor*b + pair*x*b + cubic*x**3 - v\\n        if np.max(np.abs(residual), initial=0) < 1e-11:\\n            break\\n        jac = np.zeros((len(x), 4, 4))\\n        jac[:, j, j] = lin + pair*b + 3*cubic*x*x\\n        jac[:, j, (j+1)%4] = neighbor + pair*x\\n        x -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -0.5, 0.5)\\n    return x\\n\\n\\ndef _basis(x, k, family):\\n    if family == \'sin\':\\n        h = np.sin(k*x)\\n    elif family == \'tanh\':\\n        h = np.tanh(k*x)\\n    else:\\n        h = x*np.exp(-(k*x)**2)\\n    w = np.column_stack((x, h))\\n    return np.column_stack((h, *(w[:, i]*w[:, j]\\n                                for i, j in combinations_with_replacement(range(w.shape[1]), 2))))\\n\\n\\ndef _readout(a, b, ref, reft, response, scale, families):\\n    n = len(a)\\n    baseline = np.linalg.lstsq(a, response, rcond=None)[0]\\n    residual = response - a @ baseline\\n    direction = np.linalg.svd(residual, full_matrices=False)[2][0]\\n    prediction = 0.\\n    for family, factor, penalty, weight in families:\\n        features = _basis(np.vstack((ref, reft)), scale*factor, family)\\n        features /= np.maximum(features[:n].std(axis=0), 1e-3)\\n        loading = np.linalg.lstsq(a, features[:n], rcond=None)[0]\\n        fa = features[:n] - a @ loading\\n        fb = features[n:] - b @ loading\\n        coef = np.linalg.solve(fa.T @ fa + penalty*np.eye(fa.shape[1]), fa.T @ (residual @ direction))\\n        prediction += weight*(fb @ coef)[:, None]*direction\\n    return b @ baseline + prediction\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study, latent = inputs[\'__input__\'], inputs[\'node_00\']\\n    train, test, panel = study[\'train\'], study[\'test\'], study[\'panel_a\']\\n    coef, seed = _fit_a(panel[\'u\'], panel[\'v\'])\\n    ref = _invert_a(train[\'x\'], coef, seed)\\n    reft = _invert_a(test[\'x\'], coef, seed)\\n    a = _affine(np.column_stack((latent[\'z_tr\'], ref)))\\n    b = _affine(np.column_stack((latent[\'z_te\'], reft)))\\n    response = np.column_stack((train[\'y\'], train[\'traj\']))\\n    families = [(\'sin\', 1.2, 0.03, 0.5), (\'exp\', 0.6, 0.015, 0.5)]\\n    prediction = _readout(a, b, ref, reft, response, study[\'telemetry_vec\'][0], families)\\n    y_hat, state = prediction[:, 0], prediction[:, 1:]\\n    final = state[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])}\\n"}'

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
