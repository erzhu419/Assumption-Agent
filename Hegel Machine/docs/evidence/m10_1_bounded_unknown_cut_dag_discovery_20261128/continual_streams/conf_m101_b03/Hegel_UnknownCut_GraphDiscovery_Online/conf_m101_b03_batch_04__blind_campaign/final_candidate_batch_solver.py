"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["node_00"], "node_03": ["__input__", "node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "2af6b1aa4dc4b8fc7d3bb8ff96d355489289d2ccd619792a640da0a0d60f3908", "node_01": "24572019cfcf69e5e31ad7a5dd7d0ee8d51950bab71b8685a101f94fffe62117", "node_02": "20bc11707c0e213037e7fe1caf5cfce240f415bbbec5ddfb43c2a1ae3d894523", "node_03": "e686b65cbbec9d1b79e21504dc56345feb1ba23908122276e7b5163e8c12bb43"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _latent(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    linear = np.column_stack((np.ones(len(u)), u))\\n    keep = np.ones(len(u), dtype=bool)\\n    for _ in range(8):\\n        coef = np.linalg.lstsq(linear[keep], v[keep], rcond=None)[0]\\n        error = np.max(abs(linear @ coef - v), axis=1)\\n        keep = error < max(0.001, 3 * np.median(error))\\n    u, v = u[keep], v[keep]\\n    i = np.arange(4)\\n    j = (i + 1) % 4\\n    coef = np.column_stack([\\n        np.linalg.lstsq(np.column_stack((np.ones(len(u)), u[:, a], u[:, b],\\n                                         u[:, a]*u[:, b], u[:, a]**3)),\\n                        v[:, a], rcond=None)[0] for a, b in zip(i, j)\\n    ])\\n    shift, gain, cross, product, cubic = coef\\n    z = (x-shift)/gain\\n    for _ in range(20):\\n        zj = z[:, j]\\n        error = shift + gain*z + cross*zj + product*z*zj + cubic*z**3 - x\\n        if np.max(np.abs(error), initial=0.) < 1e-12:\\n            break\\n        jac = np.zeros((len(z), 4, 4))\\n        jac[:, i, i] = gain + product*zj + 3*cubic*z*z\\n        jac[:, i, j] = cross + product*z\\n        z -= np.linalg.solve(jac, error[..., None])[..., 0]\\n    return z\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    z = _latent(study[\'panel_a\'], x)\\n    return {\'z\': z, \'panel_c\': study[\'panel_c\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _design(x):\\n    # The panel readout mixes the four adjacent-coordinate products.\\n    adjacent_products = x * np.roll(x, -1, axis=1)\\n    return np.column_stack((np.ones(len(x)), adjacent_products))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    panel = upstream[\'panel_c\']\\n    coef = np.linalg.lstsq(_design(panel[\'u\']), panel[\'v1\'], rcond=None)[0]\\n    return {\'a\': _design(upstream[\'z\']) @ coef}\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    panel, z = upstream[\'panel_c\'], upstream[\'z\']\\n    u, v = panel[\'u\'], panel[\'v2\']\\n\\n    def nonlinear(x, h):\\n        return x*x + h[1]*(np.cos(h[0]*x)-1)\\n\\n    starts = [[w, -.2] for w in (1, 2.3, 3)]\\n\\n    def design(x, h):\\n        return np.column_stack((np.ones(len(x)), x, nonlinear(x, h)))\\n\\n    def residual(h, loo=False):\\n        q, _ = np.linalg.qr(design(u, h))\\n        r = v-q@(q.T@v)\\n        if loo:\\n            r = r/(1-(q*q).sum(axis=1, keepdims=True))\\n        return r.ravel()\\n\\n    fits = (least_squares(residual, start, max_nfev=300) for start in starts)\\n    h = min(fits, key=lambda fit: np.linalg.norm(residual(fit.x, True))).x\\n    return {\'b\': design(z, h) @ np.linalg.lstsq(design(u, h), v, rcond=None)[0]}\\n", "node_03": "import numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _time_modes(times, trajectory):\\n    def basis(log_rates):\\n        return -np.expm1(-times[:, None]*np.exp(log_rates))\\n\\n    def amplitudes(h):\\n        return np.linalg.lstsq(h, trajectory.T, rcond=None)[0].T\\n\\n    def residual(log_rates):\\n        h = basis(log_rates)\\n        return (amplitudes(h) @ h.T - trajectory).ravel()\\n\\n    result = least_squares(residual, np.log([.45, 1.8]), bounds=(np.log([.05, .9]), np.log([.9, 6.])))\\n    h = basis(result.x)\\n    return h, amplitudes(h)\\n\\n\\ndef _matern(a, b):\\n    d = np.sqrt(5*np.sum((a[:, None, :] - b[None, :, :])**2, axis=2))/1.4\\n    return (1+d+d*d/3)*np.exp(-d)\\n\\n\\ndef _predict_amplitudes(features, coordinates, amplitudes):\\n    n = len(amplitudes)\\n    d = features[:n]\\n    cov = _matern(coordinates[:n], coordinates[:n]) + .001*np.eye(n)\\n    inverse_design, inverse_targets = np.split(\\n        cho_solve(cho_factor(cov), np.column_stack((d, amplitudes))),\\n        [d.shape[1]], axis=1)\\n    coef = np.linalg.solve(d.T @ inverse_design + .0003*np.eye(d.shape[1]),\\n                           d.T @ inverse_targets)\\n    weights = inverse_targets - inverse_design @ coef\\n    return features[n:] @ coef + _matern(coordinates[n:], coordinates[:n]) @ weights\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    a, b = inputs[\'node_01\'][\'a\'], inputs[\'node_02\'][\'b\']\\n    paired = a*b\\n    scalar = np.column_stack((np.ones(len(a)), a, b, paired.sum(axis=1)))\\n    state = np.column_stack((scalar[:, :-1], paired))\\n    h, amplitudes = _time_modes(study[\'eval_times\'], train[\'traj\'])\\n    y_hat = scalar[n:] @ np.linalg.lstsq(scalar[:n], train[\'y\'], rcond=None)[0]\\n    state_pred = _predict_amplitudes(state, b, amplitudes) @ h.T\\n    endpoint = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()])}\\n"}'

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
