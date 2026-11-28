"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "50e9b22646f77a8232464618556e952276106875546ea9baa6898fffbda6de6e"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover the latent coordinates and fit scalar and relaxation readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    neighbor = np.roll(u, -1, axis=1)\\n    # Channel-specific intercepts and gains; shared cyclic and cubic terms.\\n    indicators = np.broadcast_to(np.eye(4), (len(u), 4, 4))\\n    design = np.concatenate((indicators, indicators * u[..., None],\\n                             np.stack((neighbor, u * neighbor, u**3), axis=-1)), axis=-1)\\n    matrix = design.reshape(-1, design.shape[-1])\\n    start = _fit(matrix, v.ravel())\\n    robust = least_squares(lambda w: matrix @ w - v.ravel(), start,\\n                           jac=lambda w: matrix.copy(), loss=\'cauchy\', f_scale=.01).x\\n    error = np.linalg.norm(design @ robust - v, axis=1)\\n    keep = error < 8 * np.median(error)\\n    return _fit(design[keep].reshape(-1, design.shape[-1]), v[keep].ravel())\\n\\n\\ndef _invert(raw, coefficients):\\n    offset, gain = coefficients[:4], coefficients[4:8]\\n    coupling, interaction, cubic = coefficients[8:]\\n    z = (raw - offset) / gain\\n    j = np.arange(4)\\n    jacobian = np.zeros((len(z), 4, 4))\\n    for _ in range(15):\\n        neighbor = np.roll(z, -1, axis=1)\\n        error = offset + gain*z + coupling*neighbor + interaction*z*neighbor + cubic*z**3 - raw\\n        if np.max(np.abs(error), initial=0.) < 1e-12:\\n            break\\n        jacobian[:, j, j] = gain + interaction*neighbor + 3*cubic*z*z\\n        jacobian[:, j, (j+1) % 4] = coupling + interaction*z\\n        z -= np.linalg.solve(jacobian, error[..., None])[..., 0]\\n    return z\\n\\n\\ndef _scalar(b):\\n    return np.column_stack((_affine(b), b[:, 0]*b[:, 2], b[:, 1]*b[:, 3]))\\n\\n\\ndef _trajectory(x, xt, times, observations):\\n    # Fit the temporal modes to the affine spatial signal, not observation noise.\\n    q, r = np.linalg.qr(_affine(x), mode=\'reduced\')\\n    signal = q.T @ observations\\n\\n    def modes(rates):\\n        return -np.expm1(-times[:, None] * rates)\\n\\n    def residual(rates):\\n        basis = modes(rates)\\n        return (basis @ _fit(basis, signal.T) - signal.T).ravel()\\n\\n    rates = least_squares(residual, [1.7, .4],\\n                          bounds=([.7, .02], [5., .68])).x\\n    basis = np.linalg.qr(modes(rates), mode=\'reduced\')[0]\\n    return (_affine(xt) @ np.linalg.solve(r, signal @ basis)) @ basis.T\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    calibration = _calibration(study[\'panel_a\'])\\n    x = _invert(train[\'x\'], calibration)\\n    xt = _invert(study[\'test\'][\'x\'], calibration)\\n    panel = study[\'panel_b\']\\n    mixing = _fit(_affine(panel[\'u\']), panel[\'v\'])\\n    b, bt = _affine(x) @ mixing, _affine(xt) @ mixing\\n    y_hat = _scalar(bt) @ _fit(_scalar(b), train[\'y\'])\\n    state_pred = _trajectory(x, xt, study[\'eval_times\'], train[\'traj\'])\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
