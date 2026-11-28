"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05_r00", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["node_00"], "node_03": ["__input__", "node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "0e78824ca0d1ce4df5a27d80116acff066591cf788da9c40f69db30e3b99d745", "node_01": "92328e2489289e8008586e8c359900063eda94ba11a00a1481f2916a8ba7a121", "node_02": "bbaa4c8e29ed0184d1442200b66892867e0376ef1a2e49616015d047ed8783ea", "node_03": "53e0e7678726f3908133a9657bee545d479c34cded4fdb8cdab159068405d539"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit_panel_a(u, v):\\n    neighbor = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1).transpose(1, 0, 2)\\n\\n    def fit(weights):\\n        w = np.sqrt(weights)\\n        return np.stack([np.linalg.lstsq(d * w[:, None], y * w, rcond=None)[0]\\n                         for d, y in zip(design, v.T)])\\n\\n    weights = np.ones(len(u))\\n    for _ in range(12):\\n        coef = fit(weights)\\n        residual = np.linalg.norm(v - np.einsum(\'ijk,ik->ji\', design, coef), axis=1)\\n        scale = max(np.median(residual), 1e-4)\\n        weights = 1.0 / (1.0 + (residual / scale)**2)\\n    # The calibration panel contains three grossly corrupted rows.\\n    weights[:] = 1.0\\n    weights[np.argsort(residual)[-3:]] = 0.0\\n    return fit(weights)\\n\\n\\ndef _invert_panel_a(x, coef):\\n    bias, linear, cross, product, cubic = coef.T\\n    idx = np.arange(len(bias))\\n    next_idx = (idx + 1) % len(idx)\\n    affine = np.diag(linear)\\n    affine[idx, next_idx] = cross\\n    z = np.linalg.solve(affine, (x - bias).T).T\\n\\n    def residual(q):\\n        neighbor = q[:, next_idx]\\n        return bias + linear*q + (cross + product*q)*neighbor + cubic*q**3 - x\\n\\n    for _ in range(30):\\n        r = residual(z)\\n        error = np.max(np.abs(r), axis=1)\\n        if error.max() < 1e-12:\\n            break\\n        jac = np.broadcast_to(affine, (len(z), *affine.shape)).copy()\\n        jac[:, idx, idx] += product*z[:, next_idx] + 3*cubic*z*z\\n        jac[:, idx, next_idx] += product*z\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        trial = z - step\\n        for _ in range(12):\\n            worse = np.max(np.abs(residual(trial)), axis=1) > error\\n            if not np.any(worse):\\n                break\\n            step[worse] *= 0.5\\n            trial[worse] = z[worse] - step[worse]\\n        z = trial\\n    return z\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, c = study[\'panel_a\'], study[\'panel_c\']\\n    coef = _fit_panel_a(a[\'u\'], a[\'v\'])\\n    x = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']), axis=0)\\n    z = _invert_panel_a(x, coef)\\n    return {\'z\': z, \'u\': c[\'u\'], \'v1\': c[\'v1\'], \'v2\': c[\'v2\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _basis(x):\\n    # The v1 readout is affine in the products between {0, 2} and {1, 3}.\\n    x0, x1, x2, x3 = x.T\\n    return np.column_stack((np.ones(len(x)), x0*x1, x0*x3, x2*x1, x2*x3))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    coefficients = np.linalg.lstsq(_basis(upstream[\'u\']), upstream[\'v1\'], rcond=None)[0]\\n    return {\'a\': _basis(upstream[\'z\']) @ coefficients}\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    prefix = inputs[\'node_00\']\\n    u, v = prefix[\'u\'], prefix[\'v2\']\\n\\n    def basis(z, p):\\n        return z + p*z*z\\n\\n    def fit(p):\\n        f = basis(u, p)\\n        mean = f.mean(axis=0)\\n        g = f - mean\\n        left, singular, right = np.linalg.svd(g.T @ (v - v.mean(axis=0)))\\n        mat = (left @ right) * (singular.sum() / np.sum(g*g))\\n        bias = v.mean(axis=0) - mean @ mat\\n        return mat, bias\\n\\n    def error(p):\\n        mat, bias = fit(p)\\n        return np.sum((basis(u, p) @ mat + bias - v)**2)\\n\\n    p = minimize_scalar(error, bounds=(0.05, 1), method=\'bounded\').x\\n    mat, bias = fit(p)\\n    return {\'b\': basis(prefix[\'z\'], p) @ mat + bias}\\n", "node_03": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ridge(x, y, penalty):\\n    gram = x.T @ x\\n    gram[np.diag_indices_from(gram)] += penalty\\n    gram[0, 0] -= penalty\\n    return np.linalg.solve(gram, x.T @ y)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    a, b = inputs[\'node_01\'][\'a\'], inputs[\'node_02\'][\'b\']\\n    mixed = (a[:, :, None] * b[:, None, :]).reshape(len(a), 16)\\n    linear = np.column_stack((np.ones(len(a)), a, b))\\n    dot = mixed[:, [0, 5, 10, 15]].sum(axis=1)\\n    common = np.column_stack((linear, dot))\\n    slow = np.column_stack((linear, mixed[:, 1]))\\n    times = np.asarray(study[\'eval_times\'])[:, None]\\n\\n    def time_basis(rates):\\n        return -np.expm1(-times * rates)\\n\\n    def residual(rates):\\n        basis = time_basis(rates)\\n        observed = train[\'traj\'].T\\n        return (basis @ np.linalg.lstsq(basis, observed, rcond=None)[0] - observed).ravel()\\n\\n    rates = least_squares(residual, [1.8, 0.4], bounds=([1.0, 0.01], [10.0, 1.0])).x\\n    temporal = time_basis(rates)\\n    amplitudes = np.linalg.lstsq(temporal, train[\'traj\'].T, rcond=None)[0].T\\n    main = common[n:] @ _ridge(common[:n], np.column_stack((train[\'y\'], amplitudes[:, 0])), 0.00015)\\n    tail = slow[n:] @ _ridge(slow[:n], amplitudes[:, 1], 0.0003)\\n    y = main[:, 0]\\n    trajectory = np.column_stack((main[:, 1], tail)) @ temporal.T\\n    end = trajectory[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': trajectory,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
