"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "69b34e0cf80e5c06b7aa7cc3aeb5e0eec0a61300b7edeb52287aa6178d927098"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrate inputs, recover the panel-B warp, and predict sparse readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\nfrom scipy.special import erf\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _linear(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(panel):\\n    design, target = _linear(panel[\'u\']), panel[\'v\']\\n    initial = _fit(design, target)\\n    robust = least_squares(\\n        lambda b: (design @ b.reshape(initial.shape) - target).ravel(),\\n        initial.ravel(), loss=\'cauchy\', f_scale=.02,\\n    ).x.reshape(initial.shape)\\n    residual = np.linalg.norm(design @ robust - target, axis=1)\\n    keep = residual <= 10 * np.median(residual)\\n    return _fit(design[keep], target[keep])\\n\\n\\ndef _warp(x, parameters):\\n    strength, scale = parameters\\n    return _linear(x - strength * erf(scale * x))\\n\\n\\ndef _panel_operator(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def residual(parameters):\\n        design = _warp(u, parameters)\\n        return (design @ _fit(design, v) - v).ravel()\\n\\n    parameters = least_squares(residual, [.68, 1.45],\\n                               bounds=([.02, .15], [4., 4.])).x\\n    return parameters, _fit(_warp(u, parameters), v)\\n\\n\\ndef _readouts(b):\\n    b0, b1, b2, b3 = b.T\\n    scalar = np.column_stack((_linear(b), b0 * b2, b1 * b3))\\n    trajectory = np.column_stack((scalar, b0 * b3))\\n    return scalar, trajectory\\n\\n\\ndef _time_basis(times, trajectory):\\n    def basis(rates):\\n        return -np.expm1(-times[:, None] * rates)\\n\\n    def residual(rates):\\n        b = basis(rates)\\n        return (b @ _fit(b, trajectory.T) - trajectory.T).ravel()\\n\\n    rates = least_squares(residual, [1.6, .3],\\n                          bounds=([.58, .01], [5., 1.2])).x\\n    return np.linalg.qr(basis(rates), mode=\'reduced\')[0]\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    calibration = _calibration(study[\'panel_a\'])\\n    parameters, mixing = _panel_operator(study[\'panel_b\'])\\n\\n    def design(raw):\\n        calibrated = np.linalg.solve(calibration[1:].T,\\n                                     (raw - calibration[0]).T).T\\n        return _readouts(_warp(calibrated, parameters) @ mixing)\\n\\n    scalar, trajectory = design(train[\'x\'])\\n    scalar_test, trajectory_test = design(study[\'test\'][\'x\'])\\n    time_basis = _time_basis(study[\'eval_times\'], train[\'traj\'])\\n    y_hat = scalar_test @ _fit(scalar, train[\'y\'])\\n    state_pred = trajectory_test @ _fit(trajectory, train[\'traj\'] @ time_basis) @ time_basis.T\\n    final = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])\\n    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics)\\n"}'

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
