"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "ca51e804c46966d5c1a0e0ce18a893d4a5d56491ce1ed5f25a99320dbc7a235a", "node_01": "918cc73ffe65c63a713176af532c636e67b2f56dc7b672f93544009d25c19cb3", "node_02": "0bf57d98f401fb8541638d3a778e6b4171e21ab0c4532e10a402b5c2ed660d31"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit_forward(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    affine = np.column_stack((np.ones(len(u)), u))\\n    weights = np.ones(len(u))\\n    for _ in range(30):\\n        coeff = np.linalg.lstsq(affine * weights[:, None], v * weights[:, None], rcond=None)[0]\\n        residual = np.linalg.norm(affine @ coeff - v, axis=1)\\n        scale = max(2 * np.median(residual), 1e-12)\\n        weights = 1 / np.sqrt(1 + (residual / scale) ** 2)\\n\\n    # The three corrupted reference responses are row outliers.\\n    keep = np.argsort(residual)[:-3]\\n    a = u[keep]\\n    b = np.roll(a, -1, axis=1)\\n    eye = np.broadcast_to(np.eye(a.shape[1]), (len(a), a.shape[1], a.shape[1]))\\n    design = np.concatenate((eye, a[..., None] * eye,\\n                             b[..., None], (a * b)[..., None], (a ** 3)[..., None]), axis=2)\\n    p = np.linalg.lstsq(design.reshape(a.size, -1), v[keep].ravel(), rcond=None)[0]\\n    n = a.shape[1]\\n    return p[:n], p[n:2*n], p[-3], p[-2], p[-1]\\n\\n\\ndef _forward(a, params):\\n    offset, linear, neighbor, product, cubic = params\\n    b = np.roll(a, -1, axis=-1)\\n    return offset + linear*a + neighbor*b + product*a*b + cubic*a**3\\n\\n\\ndef _invert(observed, params):\\n    offset, linear, neighbor, product, cubic = params\\n    i = np.arange(len(linear))\\n    j = (i + 1) % len(i)\\n    affine = np.diag(linear)\\n    affine[i, j] = neighbor\\n    initial = np.linalg.solve(affine, (observed - offset).T).T\\n\\n    def jacobian(a):\\n        jac = np.diag(linear + product*a[j] + 3*cubic*a*a)\\n        jac[i, j] = neighbor + product*a\\n        return jac\\n\\n    return np.stack([least_squares(lambda a: _forward(a, params) - row, guess,\\n                                   jac=jacobian, gtol=1e-12).x\\n                     for row, guess in zip(observed, initial)])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    observed = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']), axis=0)\\n    params = _fit_forward(study[\'panel_a\'])\\n    return {\'calibrated_x\': _invert(observed, params), \'panel_b\': study[\'panel_b\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _affine_features(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    panel = upstream[\'panel_b\']\\n    operator = np.linalg.lstsq(_affine_features(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    return {\'x\': _affine_features(upstream[\'calibrated_x\']) @ operator}\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\nfrom scipy.special import hyp2f1\\n\\n\\ndef _basis(times, rates):\\n    return -np.expm1(-np.exp(rates)[:, None] * times)\\n\\n\\ndef _time_fit(times, traj):\\n    def residual(rates):\\n        b = _basis(times, rates)\\n        return (traj @ np.linalg.pinv(b) @ b - traj).ravel()\\n    fit = least_squares(residual, [-1.0, 1.0], bounds=(-8.0, 4.0))\\n    return _basis(times, np.sort(fit.x))\\n\\n\\ndef _features(x, p):\\n    a = x * np.exp(p[0]) + p[1:5]\\n    h = a * hyp2f1(0.5, p[5], 1.5, -a*a)\\n    state = np.column_stack((np.ones(len(x)), h, h*h))\\n    left, right = np.triu_indices(4, 1)\\n    readout = np.column_stack((state, h[:, left] * h[:, right]))\\n    return readout, state\\n\\n\\ndef _spatial_fit(x, targets):\\n    scale = np.maximum(0.05, targets.std(axis=0))\\n    normalized = targets / scale\\n    def profile(d, v):\\n        penalty = np.diag(np.r_[0., np.full(d.shape[1] - 1, 0.02)])\\n        w = np.linalg.lstsq(np.vstack((d, penalty)),\\n                            np.vstack((v, np.zeros((len(penalty), v.shape[1])))),\\n                            rcond=None)[0]\\n        return w, (d @ w - v).ravel(), (penalty @ w).ravel()\\n\\n    def residual(p):\\n        d, g = _features(x, p)\\n        _, e1, r1 = profile(d, normalized[:, :1])\\n        _, e2, r2 = profile(g, normalized[:, 1:])\\n        return np.r_[e1, e2, r1, r2, 0.07*p[1:5]]\\n\\n    fit = least_squares(residual, [np.log(3.), 0., 0., 0., 0., 0.5],\\n                        bounds=([-1.5] + [-0.45]*4 + [0.2], [1.9] + [0.45]*4 + [1.3]),\\n                        max_nfev=75)\\n    d, g = _features(x, fit.x)\\n    w, _, _ = profile(d, normalized[:, :1])\\n    v, _, _ = profile(g, normalized[:, 1:])\\n    return fit.x, w[:, 0] * scale[0], v * scale[1:]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    x = inputs[\'node_01\'][\'x\']\\n    n = len(train[\'x\'])\\n    b = _time_fit(study[\'eval_times\'], train[\'traj\'])\\n    targets = np.column_stack((train[\'y\'], train[\'traj\'] @ np.linalg.pinv(b)))\\n    p, w, v = _spatial_fit(x[:n], targets)\\n    d, g = _features(x[n:], p)\\n    y, state = d @ w, g @ v @ b\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}\\n"}'

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
