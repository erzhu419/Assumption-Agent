"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "36c6ae32746a7694cabb65870eb5c38b776d637d842dfd86731db5db8b257179", "node_01": "89eb88a06179f94e839e69ceb190f71c007c6481b03f40ed65f0cdaa8a04af6b"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit_panel_a(u, v):\\n    nxt = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, nxt, u*nxt, u**3), axis=1)\\n\\n    def fit(w):\\n        return np.stack([np.linalg.lstsq(design[:, :, j]*w[:, None], v[:, j]*w, rcond=None)[0] for j in range(u.shape[1])], axis=1)\\n\\n    weights = np.ones(len(u))\\n    for _ in range(25):\\n        coef = fit(weights)\\n        residual = np.linalg.norm(np.einsum(\'nkj,kj->nj\', design, coef) - v, axis=1)\\n        scale = max(0.002, np.median(residual))\\n        weights = scale / np.hypot(scale, residual)\\n    keep = np.argsort(residual)[:-3]\\n    weights = np.zeros(len(u))\\n    weights[keep] = 1.0\\n    return fit(weights)\\n\\n\\ndef _forward_a(z, coef):\\n    offset, linear, neighbor, interaction, cubic = coef\\n    nxt = np.roll(z, -1, axis=1)\\n    return offset + linear*z + (neighbor + interaction*z)*nxt + cubic*z**3\\n\\n\\ndef _inverse_a(v, coef):\\n    offset, linear, neighbor, interaction, cubic = coef\\n    z = np.zeros_like(v)\\n    i = np.arange(v.shape[1])\\n    for _ in range(35):\\n        residual = _forward_a(z, coef) - v\\n        if np.max(np.abs(residual)) < 1e-11:\\n            break\\n        jac = np.zeros((len(v), len(i), len(i)))\\n        jac[:, i, i] = linear + interaction*np.roll(z, -1, axis=1) + 3*cubic*z*z\\n        jac[:, i, (i+1) % len(i)] = neighbor + interaction*z\\n        step = np.linalg.solve(jac, residual[..., None])[..., 0]\\n        err = np.linalg.norm(residual, axis=1)\\n        damping = np.ones(len(z))\\n        for _ in range(15):\\n            trial = z - damping[:, None]*step\\n            bad = np.linalg.norm(_forward_a(trial, coef)-v, axis=1) > err\\n            if not np.any(bad):\\n                break\\n            damping[bad] *= 0.5\\n        z = trial\\n    return z\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    panel_a = study[\'panel_a\']\\n    panel_b = study[\'panel_b\']\\n    coef_a = _fit_panel_a(panel_a[\'u\'], panel_a[\'v\'])\\n    design_b = np.column_stack((np.ones(len(panel_b[\'u\'])), panel_b[\'u\']))\\n    offset_b, *linear_rows_b = np.linalg.lstsq(design_b, panel_b[\'v\'], rcond=None)[0]\\n    linear_b = np.asarray(linear_rows_b)\\n\\n    # Panel A maps into observation space; Panel B maps into sink space.\\n    train_x = np.asarray(study[\'train\'][\'x\'], dtype=float)\\n    test_x = np.asarray(study[\'test\'][\'x\'], dtype=float)\\n    observed = np.concatenate((train_x, test_x), axis=0)\\n    z = _inverse_a(observed, coef_a) @ linear_b + offset_b\\n    return {\'z_tr\': z[:len(train_x)], \'z_te\': z[len(train_x):]}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _scalar_design(z):\\n    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]\\n    return np.column_stack((z, interaction))\\n\\n\\ndef _trajectory(x, target, x_test, t):\\n    def basis(rates):\\n        return -np.expm1(-t[:, None] * rates)\\n\\n    def residual(rates):\\n        b = basis(rates)\\n        return (b @ _fit(b, target.T) - target.T).ravel()\\n\\n    rates = least_squares(residual, [0.4, 1.8], bounds=([0.01, 0.8], [0.8, 8.0])).x\\n    b = basis(rates)\\n    return (x_test @ _fit(x, _fit(b, target.T).T)) @ b.T\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    coords = inputs[\'node_00\']\\n    x, x_test = coords[\'z_tr\'], coords[\'z_te\']\\n    y_hat = _scalar_design(x_test) @ _fit(_scalar_design(x), train[\'y\'])\\n    state_pred = _trajectory(x, train[\'traj\'], x_test, study[\'eval_times\'])\\n    end = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
