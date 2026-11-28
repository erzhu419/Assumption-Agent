"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "e3b0c26feac2e91fe7fb1fb8e48e3a9696756101eac24f479b352c83858a44a2"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Predict the scalar response and two-mode trajectory from calibrated operators.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=1e-9)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _pairs(z):\\n    return z[:, [0, 0, 1, 2]] * z[:, [1, 3, 2, 3]]\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    # Coordinatewise affine calibration. Joint row trimming rejects panel outliers.\\n    keep = np.ones(len(u), dtype=bool)\\n    for _ in range(8):\\n        uc, vc = u[keep], v[keep]\\n        centered = uc - uc.mean(axis=0)\\n        gain = np.sum(centered * (vc - vc.mean(axis=0)), axis=0) / np.sum(centered**2, axis=0)\\n        shift = vc.mean(axis=0) - uc.mean(axis=0) * gain\\n        error = np.linalg.norm(u * gain + shift - v, axis=1)\\n        updated = error <= 4 * np.median(error)\\n        if np.array_equal(keep, updated):\\n            break\\n        keep = updated\\n    return (x - shift) / gain\\n\\n\\ndef _features(study, x):\\n    z = _calibrate(study[\'panel_a\'], x)\\n    pb, pc = study[\'panel_b\'], study[\'panel_c\']\\n    b = _affine(z) @ _fit(_affine(pb[\'u\']), pb[\'v\'])\\n    q = _pairs(z)\\n    c = _affine(q) @ _fit(_affine(_pairs(pc[\'u\'])), pc[\'v1\'])\\n\\n    def cosine_design(rate):\\n        return _affine(np.cos(rate * pc[\'u\']))\\n\\n    def residual(rate):\\n        f = cosine_design(rate)\\n        return np.linalg.norm(f @ _fit(f, pc[\'v2\']) - pc[\'v2\'])\\n\\n    rate = minimize_scalar(residual, bounds=(2.2, 2.6), method=\'bounded\').x\\n    h = np.cos(rate * z)\\n    d = _affine(h) @ _fit(cosine_design(rate), pc[\'v2\'])\\n    return z, b, q, c, h, d\\n\\n\\ndef _dot(a, b, shift=0):\\n    return np.sum(a * np.roll(b, shift, axis=1), axis=1)\\n\\n\\ndef _designs(study, x):\\n    z, b, q, c, h, d = _features(study, x)\\n    scalar = np.column_stack((_affine(q), h, _dot(c, d)))\\n    mixed = (q[:, :, None] * h[:, None, :]).reshape(len(x), 16)\\n    state = np.column_stack((_affine(q), h, _dot(b, c, 2), _dot(z, c), mixed))\\n    return scalar, state\\n\\n\\ndef _time_modes(times, trajectory):\\n    def basis(rates):\\n        return -np.expm1(-times[:, None] * rates)\\n\\n    def residual(rates):\\n        h = basis(rates)\\n        return (h @ _fit(h, trajectory.T) - trajectory.T).ravel()\\n\\n    rates = least_squares(residual, [.5, 2.], bounds=(.01, 10.)).x\\n    return basis(np.sort(rates))\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    x = np.concatenate((train[\'x\'], study[\'test\'][\'x\']))\\n    scalar, state = _designs(study, x)\\n    h = _time_modes(study[\'eval_times\'], train[\'traj\'])\\n    amps = _fit(h, train[\'traj\'].T).T\\n    y_hat = scalar[n:] @ _fit(scalar[:n], train[\'y\'])\\n    state_pred = (state[n:] @ _fit(state[:n], amps)) @ h.T\\n    endpoint = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
