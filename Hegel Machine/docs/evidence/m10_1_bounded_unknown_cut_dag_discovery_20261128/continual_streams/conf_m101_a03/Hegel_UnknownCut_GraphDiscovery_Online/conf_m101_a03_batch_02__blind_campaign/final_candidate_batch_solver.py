"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "c61d061175d59336da51488a7bd5aa1578426f62544c368c414cc690caaf770a", "node_01": "7256b45ec98c55c07cbfdf46c46cc07d4ec3ff082cc0ab990e55986833dbb491", "node_02": "d719d94eb32349976c0371069c474dce64f826839c9a6d24968af2d3146352f3"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _basis(u):\\n    neighbor = np.roll(u, -1, axis=1)\\n    return np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)\\n\\n\\ndef _fit_forward(u, v):\\n    # Use one weight per panel row because the contamination is row-wise.\\n    design = _basis(u)\\n    weight = np.ones(len(u))\\n\\n    def fit(w):\\n        return np.stack([\\n            np.linalg.lstsq(design[:, j] * w[:, None], v[:, j] * w, rcond=None)[0]\\n            for j in range(u.shape[1])\\n        ])\\n\\n    for _ in range(16):\\n        coeff = fit(weight)\\n        residual = np.linalg.norm(np.einsum(\'njk,jk->nj\', design, coeff) - v, axis=1)\\n        scale = max(np.median(residual), 1e-8)\\n        weight = 1.0 / np.hypot(1.0, residual / (2.0 * scale))\\n    return fit(weight)\\n\\n\\ndef _invert(v, coeff):\\n    if not len(v):\\n        return v.copy()\\n    bias, gain, cross, interaction, cubic = coeff.T\\n    j = np.arange(len(bias))\\n    k = (j + 1) % len(j)\\n    linear = np.diag(gain)\\n    linear[j, k] = cross\\n    u = np.linalg.solve(linear, (v - bias).T).T\\n\\n    def error(x):\\n        neighbor = x[:, k]\\n        return bias + gain * x + cross * neighbor + interaction * x * neighbor + cubic * x**3 - v\\n\\n    residual = error(u)\\n    jac = np.zeros((len(v), len(j), len(j)))\\n    for _ in range(30):\\n        if np.max(np.abs(residual)) < 1e-11:\\n            break\\n        jac[:, j, j] = gain + interaction * u[:, k] + 3 * cubic * u**2\\n        jac[:, j, k] = cross + interaction * u\\n        step = np.linalg.solve(jac, residual[..., None])[..., 0]\\n        old_norm = np.sum(residual**2, axis=1)\\n        factor = np.ones((len(v), 1))\\n        for _ in range(16):\\n            trial = u - factor * step\\n            trial_residual = error(trial)\\n            better = np.sum(trial_residual**2, axis=1) < old_norm\\n            pending = (~better) & (old_norm > 1e-22)\\n            if not np.any(pending):\\n                break\\n            factor[pending] *= 0.5\\n        u[better] = trial[better]\\n        residual[better] = trial_residual[better]\\n    return u\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    panel = study[\'panel_a\']\\n    coeff = _fit_forward(np.asarray(panel[\'u\'], dtype=float), np.asarray(panel[\'v\'], dtype=float))\\n    train = np.asarray(study[\'train\'][\'x\'], dtype=float)\\n    test = np.asarray(study[\'test\'][\'x\'], dtype=float)\\n    calibrated = _invert(np.concatenate((train, test)), coeff)\\n    return {\'x_tr_cal\': calibrated[:len(train)], \'x_te_cal\': calibrated[len(train):]}\\n", "node_01": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    calibrated = inputs[\'node_00\']\\n    panel_b = study[\'panel_b\']\\n    operator = _fit(_affine(panel_b[\'u\']), panel_b[\'v\'])\\n    z = _affine(calibrated[\'x_tr_cal\']) @ operator\\n    z_test = _affine(calibrated[\'x_te_cal\']) @ operator\\n    train = study[\'train\']\\n    return {\'z_tr\': z, \'z_te\': z_test, \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_02": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_01\']\\n    z, z_test = features[\'z_tr\'], features[\'z_te\']\\n    a, a_test = _affine(z), _affine(z_test)\\n    y = features[\'y\']\\n\\n    i, j = np.array(list(combinations_with_replacement(range(z.shape[1]), 2))).T\\n    products = z[:, i] * z[:, j]\\n    quadratic = _fit(np.column_stack((a, products)), y)[a.shape[1]:]\\n    keep = np.argsort(np.abs(quadratic))[-2:]\\n    pooled = products[:, keep].sum(axis=1)\\n    pooled_test = (z_test[:, i[keep]] * z_test[:, j[keep]]).sum(axis=1)\\n    y_hat = np.column_stack((a_test, pooled_test)) @ _fit(np.column_stack((a, pooled)), y)\\n\\n    traj = features[\'traj\']\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    state_pred = (a_test @ _fit(a, traj @ modes.T)) @ modes\\n    last = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
