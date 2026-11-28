"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "767f7b9cd686d14543a9098a5f3f00d125009bf0f2a2901e7ebc88fe8237e8f2", "node_01": "030cd8cff8d009615054283fb13d9416428b1aa78c33bbd572a37afc771303e4"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit_panel_a(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    nxt = np.roll(u, -1, axis=1)\\n    # Each output has a linear gain, a cubic self term, and coupling to its successor.\\n    design = np.stack((np.ones_like(u), u, nxt, u * nxt, u**3), axis=-1).transpose(1, 0, 2)\\n    targets = v.T[..., None]\\n\\n    def fit(weights):\\n        return np.stack([np.linalg.lstsq(d * weights[:, None], y * weights[:, None], rcond=None)[0][:, 0]\\n                         for d, y in zip(design, targets)])\\n\\n    weights = np.ones(len(u))\\n    for _ in range(25):\\n        coeff = fit(weights)\\n        residual = np.linalg.norm((design @ coeff[..., None])[..., 0].T - v, axis=1)\\n        scale = max(1.5 * np.median(residual), 1e-10)\\n        weights = 1 / np.hypot(1, residual / scale)\\n    # Refit without the isolated gross errors so their magnitude cannot bias the gains.\\n    return fit((residual <= 8 * scale).astype(float))\\n\\n\\ndef _invert_panel_a(x, coeff):\\n    bias, gain, cross, pair, cubic = coeff.T\\n    j = np.arange(x.shape[1])\\n    nxt = (j + 1) % len(j)\\n    linear = np.diag(gain)\\n    linear[j, nxt] = cross\\n    z = np.linalg.solve(linear, (x - bias).T).T\\n\\n    def residual(z):\\n        return bias + gain * z + (cross + pair * z) * z[:, nxt] + cubic * z**3 - x\\n\\n    for _ in range(30):\\n        r = residual(z)\\n        if np.max(np.abs(r)) < 1e-12:\\n            break\\n        jac = np.zeros((len(z), len(j), len(j)))\\n        jac[:, j, j] = gain + pair * z[:, nxt] + 3 * cubic * z**2\\n        jac[:, j, nxt] = cross + pair * z\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        new = z - step\\n        old_norm = np.sum(r**2, axis=1)\\n        for _ in range(12):\\n            worse = np.sum(residual(new)**2, axis=1) > old_norm\\n            if not worse.any():\\n                break\\n            step[worse] *= 0.5\\n            new[worse] = z[worse] - step[worse]\\n        z = new\\n    return z\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    coeff = _fit_panel_a(study[\'panel_a\'])\\n    u, v = study[\'panel_b\'][\'u\'], study[\'panel_b\'][\'v\']\\n    design = np.column_stack((np.ones(len(u)), u))\\n    affine = np.linalg.lstsq(design, v, rcond=None)[0]\\n\\n    def calibrate(x):\\n        return _invert_panel_a(x, coeff) @ affine[1:] + affine[0]\\n\\n    return {\'z_tr\': calibrate(train[\'x\']), \'z_te\': calibrate(test[\'x\']),\\n            \'y\': train[\'y\'], \'traj\': train[\'traj\'], \'eval_times\': study[\'eval_times\']}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _cross(z):\\n    return z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]\\n\\n\\ndef _relaxation(design, trajectories, times):\\n    q, r = np.linalg.qr(design, mode=\'reduced\')\\n    projected = q.T @ trajectories\\n\\n    def basis(log_rates):\\n        return -np.expm1(-times[:, None] * np.exp(log_rates))\\n\\n    def residual(log_rates):\\n        b = basis(log_rates)\\n        return (b @ _ls(b, projected.T) - projected.T).ravel()\\n\\n    result = least_squares(residual, np.log([0.4, 1.8]),\\n                           bounds=(np.log(0.03), np.log(15.0)))\\n    b = basis(result.x)\\n    weights = _ls(r, _ls(b, projected.T).T)\\n    return weights @ b.T\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    z, z_test = features[\'z_tr\'], features[\'z_te\']\\n    design, test_design = _affine(z), _affine(z_test)\\n    w = _ls(np.column_stack((z, _cross(z))), features[\'y\'])\\n    y_hat = np.column_stack((z_test, _cross(z_test))) @ w\\n    state_pred = test_design @ _relaxation(design, features[\'traj\'], features[\'eval_times\'])\\n    final_state = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(),\\n                                         final_state.mean(), final_state.std()])}\\n"}'

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
