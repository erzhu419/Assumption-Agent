"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "564b978662e4c604bb7ad2064ea0a1a4a78889000407a07fcc75e4f233cd5d4f", "node_01": "58e6c561f5404291c1a7591c8a1a0758cdfae9f359cce935f0d7bf443adbe5e7"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibrate_a(u, v):\\n    neighbor = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)\\n    w = np.ones(len(u))\\n    for _ in range(16):\\n        coef = np.stack([_fit(design[:, j, :] * w[:, None], v[:, j] * w)\\n                         for j in range(u.shape[1])], axis=1)\\n        residual = np.linalg.norm(np.einsum(\'ndk,kd->nd\', design, coef) - v, axis=1)\\n        w = 1 / np.hypot(1, residual / 0.02)\\n    # Refit after removing the three grossly contaminated calibration rows.\\n    keep = np.argsort(residual)[:-3]\\n    return np.stack([_fit(design[keep, j, :], v[keep, j])\\n                     for j in range(u.shape[1])], axis=1)\\n\\n\\ndef _invert_a(x, coef):\\n    offset, diagonal, coupling, product, cubic = coef\\n    j = np.arange(x.shape[1])\\n    k = (j + 1) % len(j)\\n    matrix = np.diag(diagonal)\\n    matrix[j, k] = coupling\\n    u = np.linalg.solve(matrix, (x - offset).T).T\\n\\n    def residual(a):\\n        b = a[..., k]\\n        return offset + diagonal * a + coupling * b + product * a * b + cubic * a**3 - x\\n\\n    for _ in range(24):\\n        r = residual(u)\\n        err = np.linalg.norm(r, axis=1)\\n        if np.max(err, initial=0) < 1e-11:\\n            break\\n        jac = np.zeros((len(u), len(j), len(j)))\\n        jac[:, j, j] = diagonal + product * u[:, k] + 3 * cubic * u**2\\n        jac[:, j, k] = coupling + product * u\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        proposal = u - step\\n        for _ in range(12):\\n            bad = np.linalg.norm(residual(proposal), axis=1) > err\\n            if not bad.any():\\n                break\\n            proposal[bad] = (u[bad] + proposal[bad]) * 0.5\\n        u = proposal\\n    return u\\n\\n\\ndef _panel_c(u, v, query, readout):\\n    def design(a, scale):\\n        if readout == 1:\\n            extra = (a / (1 + scale * np.abs(a)), a * np.abs(a))\\n        else:\\n            extra = (np.cos(scale * a) - 1, a**2)\\n        return np.column_stack((_affine(a), *extra))\\n\\n    def residual(scale):\\n        z = design(u, scale[0])\\n        return (z @ _fit(z, v) - v).ravel()\\n\\n    scale = least_squares(residual, [0.83 if readout == 1 else 1.3],\\n                          bounds=(0.25, 2.5)).x[0]\\n    return design(query, scale) @ _fit(design(u, scale), v)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    pa, pb, pc = (study[key] for key in (\'panel_a\', \'panel_b\', \'panel_c\'))\\n    x = np.concatenate((train[\'x\'], study[\'test\'][\'x\']), axis=0)\\n    latent = _invert_a(x, _calibrate_a(pa[\'u\'], pa[\'v\']))\\n    b = _affine(latent) @ _fit(_affine(pb[\'u\']), pb[\'v\'])\\n    c1 = _panel_c(pc[\'u\'], pc[\'v1\'], latent, 1)\\n    c2 = _panel_c(pc[\'u\'], pc[\'v2\'], latent, 2)\\n    return {\'n\': np.asarray(len(train[\'x\'])),\\n            \'linear\': np.column_stack((_affine(b), c1, c2)),\\n            \'mixed\': (c1[:, :, None] * c2[:, None, :]).reshape(len(x), -1),\\n            \'y\': np.asarray(train[\'y\']), \'traj\': np.asarray(train[\'traj\']),\\n            \'eval_times\': np.asarray(study[\'eval_times\'])}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ridge(x, y, penalty):\\n    gram = x.T @ x\\n    gram[np.diag_indices_from(gram)] += penalty\\n    gram[0, 0] -= penalty\\n    return np.linalg.solve(gram, x.T @ y)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    n = features[\'n\']\\n    linear, mixed = features[\'linear\'], features[\'mixed\']\\n    dot = mixed[:, [0, 5, 10, 15]].sum(axis=1)\\n    common = np.column_stack((linear, dot))\\n    slow = np.column_stack((linear, mixed[:, 1]))\\n    times = np.asarray(features[\'eval_times\'])[:, None]\\n\\n    def time_basis(rates):\\n        return -np.expm1(-times * rates)\\n\\n    def residual(rates):\\n        basis = time_basis(rates)\\n        observed = features[\'traj\'].T\\n        return (basis @ np.linalg.lstsq(basis, observed, rcond=None)[0] - observed).ravel()\\n\\n    rates = least_squares(residual, [1.8, 0.4], bounds=([1.0, 0.01], [10.0, 1.0])).x\\n    temporal = time_basis(rates)\\n    amplitudes = np.linalg.lstsq(temporal, features[\'traj\'].T, rcond=None)[0].T\\n    main = common[n:] @ _ridge(common[:n], np.column_stack((features[\'y\'], amplitudes[:, 0])), 0.00015)\\n    tail = slow[n:] @ _ridge(slow[:n], amplitudes[:, 1], 0.0003)\\n    y = main[:, 0]\\n    trajectory = np.column_stack((main[:, 1], tail)) @ temporal.T\\n    end = trajectory[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': trajectory,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
