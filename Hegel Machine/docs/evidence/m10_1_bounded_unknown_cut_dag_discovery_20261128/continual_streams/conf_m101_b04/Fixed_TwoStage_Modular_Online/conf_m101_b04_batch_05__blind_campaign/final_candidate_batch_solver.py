"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "d562a43a2ac23bc312404eb1ac495c5055e96273797f9c7b82562473fb66f655", "node_01": "8e6566e5fb146a2b1697e4d34ef467d01c9651efcfaba71edb411431b937db50"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit_panel(panel):\\n    \\"\\"\\"Fit the forward cyclic calibration on the clean reference coordinates.\\"\\"\\"\\n    u, v = panel[\'u\'], panel[\'v\']\\n    neighbor = np.roll(u, -1, axis=1)\\n    n, d = u.shape\\n    eye = np.eye(d)\\n    offsets = np.broadcast_to(eye, (n, d, d))\\n    design = np.concatenate((offsets, u[..., None] * eye,\\n                             np.stack((neighbor, u * neighbor, u**3), axis=-1)), axis=-1)\\n\\n    def fit(weights):\\n        return np.linalg.lstsq((design * weights[:, None, None]).reshape(-1, 2*d+3),\\n                               (v * weights[:, None]).ravel(), rcond=None)[0]\\n\\n    weights = np.ones(n)\\n    for _ in range(24):\\n        residual = np.linalg.norm(design @ fit(weights) - v, axis=1)\\n        scale = max(0.7 * np.median(residual), 1e-12)\\n        weights = 1 / np.hypot(1, residual / scale)\\n\\n    # Refine without shrinkage after separating the three corrupted panel rows.\\n    clean = np.ones(n)\\n    clean[np.argsort(residual)[-3:]] = 0\\n    return fit(clean)\\n\\n\\ndef _invert(x, coefficients):\\n    d = x.shape[1]\\n    bias, gain = coefficients[:d], coefficients[d:2*d]\\n    cross, interaction, cubic = coefficients[2*d:]\\n    i = np.arange(d)\\n    j = (i + 1) % d\\n    linear = np.diag(gain)\\n    linear[i, j] = cross\\n    z = np.linalg.solve(linear, (x - bias).T).T\\n    jac = np.zeros((len(x), d, d))\\n    for _ in range(25):\\n        neighbor = z[:, j]\\n        residual = bias + gain*z + cross*neighbor + interaction*z*neighbor + cubic*z**3 - x\\n        if np.max(np.abs(residual)) < 1e-12:\\n            break\\n        jac[:, i, i] = gain + interaction*neighbor + 3*cubic*z*z\\n        jac[:, i, j] = cross + interaction*z\\n        step = np.linalg.solve(jac, residual[..., None])[..., 0]\\n        z -= step\\n    return z\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    coefficients = _fit_panel(study[\'panel_a\'])\\n    calibrated = _invert(np.concatenate((train[\'x\'], test[\'x\']), axis=0), coefficients)\\n    n = len(train[\'x\'])\\n    return dict(z_tr=calibrated[:n], z_te=calibrated[n:], panel_b=study[\'panel_b\'],\\n                eval_times=study[\'eval_times\'], y=train[\'y\'], traj=train[\'traj\'])\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\nfrom scipy.special import erf\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _linear(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _warp(x, parameters):\\n    strength, scale = parameters\\n    return _linear(x - strength * erf(scale * x))\\n\\n\\ndef _panel_operator(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def residual(parameters):\\n        design = _warp(u, parameters)\\n        return (design @ _fit(design, v) - v).ravel()\\n\\n    parameters = least_squares(residual, [.68, 1.45],\\n                               bounds=([.02, .15], [4., 4.])).x\\n    return parameters, _fit(_warp(u, parameters), v)\\n\\n\\ndef _readouts(b):\\n    b0, b1, b2, b3 = b.T\\n    scalar = np.column_stack((_linear(b), b0 * b2, b1 * b3))\\n    trajectory = np.column_stack((scalar, b0 * b3))\\n    return scalar, trajectory\\n\\n\\ndef _time_basis(times, trajectory):\\n    def basis(rates):\\n        return -np.expm1(-times[:, None] * rates)\\n\\n    def residual(rates):\\n        b = basis(rates)\\n        return (b @ _fit(b, trajectory.T) - trajectory.T).ravel()\\n\\n    rates = least_squares(residual, [1.6, .3],\\n                          bounds=([.58, .01], [5., 1.2])).x\\n    return np.linalg.qr(basis(rates), mode=\'reduced\')[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    parameters, mixing = _panel_operator(data[\'panel_b\'])\\n    scalar, trajectory = _readouts(_warp(data[\'z_tr\'], parameters) @ mixing)\\n    scalar_test, trajectory_test = _readouts(_warp(data[\'z_te\'], parameters) @ mixing)\\n    time_basis = _time_basis(data[\'eval_times\'], data[\'traj\'])\\n    y_hat = scalar_test @ _fit(scalar, data[\'y\'])\\n    state_pred = trajectory_test @ _fit(trajectory, data[\'traj\'] @ time_basis) @ time_basis.T\\n    final = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])\\n    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics)\\n"}'

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
