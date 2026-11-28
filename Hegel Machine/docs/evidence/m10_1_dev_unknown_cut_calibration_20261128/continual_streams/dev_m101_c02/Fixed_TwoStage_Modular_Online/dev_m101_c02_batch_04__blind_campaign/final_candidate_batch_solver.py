"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "c3d52090225512e869f28731ab667a74ac61fe1d0b222aab3de1710ae3e66973", "node_01": "89eb88a06179f94e839e69ceb190f71c007c6481b03f40ed65f0cdaa8a04af6b"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom itertools import combinations_with_replacement\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n_TERMS = tuple(t for d in (1, 2, 3) for t in combinations_with_replacement(range(4), d))\\n\\ndef _poly(x):\\n    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, t], axis=1) for t in _TERMS])\\n\\ndef _fit_a(u, v):\\n    X = _poly(u)\\n    Q = X[:, :15]\\n    w = np.ones(len(u))\\n    for _ in range(30):\\n        beta = np.linalg.lstsq(Q*w[:, None], v*w[:, None], rcond=None)[0]\\n        r = np.linalg.norm(Q@beta-v, axis=1)\\n        w = 1/np.hypot(1, r/max(0.001, np.median(r)))\\n    keep = np.argsort(r)[:-3]\\n    return np.linalg.lstsq(X[keep], v[keep], rcond=None)[0]\\n\\n\\ndef _invert_a(v, b):\\n    x = np.linalg.solve(b[1:5].T, (v-b[0]).T).T\\n    for _ in range(30):\\n        r = _poly(x)@b-v\\n        jac = np.zeros((len(v), 4, 4))\\n        for t, c in zip(_TERMS, b[1:]):\\n            for j, k in enumerate(t):\\n                m = t[:j]+t[j+1:]\\n                jac[:, :, k] += np.prod(x[:, m], axis=1)[:, None]*c\\n        dx = np.linalg.solve(jac, r[..., None])[..., 0]\\n        if np.max(np.abs(dx)) < 1e-11:\\n            break\\n        target = np.sum(r*r, axis=1)\\n        for _ in range(12):\\n            bad = np.sum((_poly(x-dx)@b-v)**2, axis=1) > target\\n            if not np.any(bad):\\n                break\\n            dx[bad] *= 0.5\\n        x -= dx\\n    return x\\n\\ndef _basis_b(x, omega):\\n    return np.column_stack((np.ones(len(x)), x, np.sin(omega*x)))\\n\\ndef _warp_design(x, omega, gain):\\n    return np.column_stack((np.ones(len(x)), x + gain*np.sin(omega*x)))\\n\\n\\ndef _fit_b(u, v):\\n    # Initialize the shared warp from the frequency of the unrestricted harmonic fit.\\n    def profile(omega):\\n        X = _basis_b(u, omega)\\n        b = np.linalg.lstsq(X, v, rcond=None)[0]\\n        return np.sum((X@b-v)**2), b\\n    grid = np.linspace(0.5, 5, 46)\\n    w = min(grid, key=lambda s: profile(s)[0])\\n    w = minimize_scalar(lambda s: profile(s)[0], bounds=(w-0.1, w+0.1), method=\'bounded\').x\\n\\n    def shared_profile(params):\\n        X = _warp_design(u, *params)\\n        b = np.linalg.lstsq(X, v, rcond=None)[0]\\n        return (X@b-v).ravel(), b\\n\\n    params = least_squares(lambda t: shared_profile(t)[0], [w, -0.4]).x\\n    return params, shared_profile(params)[1]\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    coeff_a = _fit_a(a[\'u\'], a[\'v\'])\\n    warp, coeff_b = _fit_b(b[\'u\'], b[\'v\'])\\n    def calibrate(x):\\n        return _warp_design(_invert_a(x, coeff_a), *warp)@coeff_b\\n    return {\'z_tr\': calibrate(study[\'train\'][\'x\']), \'z_te\': calibrate(study[\'test\'][\'x\'])}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _scalar_design(z):\\n    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]\\n    return np.column_stack((z, interaction))\\n\\n\\ndef _trajectory(x, target, x_test, t):\\n    def basis(rates):\\n        return -np.expm1(-t[:, None] * rates)\\n\\n    def residual(rates):\\n        b = basis(rates)\\n        return (b @ _fit(b, target.T) - target.T).ravel()\\n\\n    rates = least_squares(residual, [0.4, 1.8], bounds=([0.01, 0.8], [0.8, 8.0])).x\\n    b = basis(rates)\\n    return (x_test @ _fit(x, _fit(b, target.T).T)) @ b.T\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    coords = inputs[\'node_00\']\\n    x, x_test = coords[\'z_tr\'], coords[\'z_te\']\\n    y_hat = _scalar_design(x_test) @ _fit(_scalar_design(x), train[\'y\'])\\n    state_pred = _trajectory(x, train[\'traj\'], x_test, study[\'eval_times\'])\\n    end = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
