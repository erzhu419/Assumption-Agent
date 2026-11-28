"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "296e7bae8a59cf3033d3d26e964d086f1e0e246facc6dd7aad1673fe06473f40", "node_01": "e9ec0a632024444877955a4651824ebe0276b45fd151c2c953f058c789c980c7"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _panel_a(panel):\\n    design, target = _affine(panel[\'u\']), panel[\'v\']\\n    coef = _fit(design, target)\\n    for _ in range(20):\\n        residual = np.linalg.norm(design @ coef - target, axis=1)\\n        scale = max(np.median(residual), 1e-8)\\n        weight = 1.0 / np.hypot(1.0, residual / scale)\\n        coef = _fit(design * weight[:, None], target * weight[:, None])\\n    residual = np.linalg.norm(design @ coef - target, axis=1)\\n    keep = np.argsort(residual)[:-3]\\n    return _fit(design[keep], target[keep])\\n\\n\\ndef _inverse(x, coef):\\n    return np.linalg.solve(coef[1:].T, (x - coef[0]).T).T\\n\\n\\ndef _logcosh(x):\\n    return np.logaddexp(x, -x) - np.log(2.0)\\n\\n\\ndef _readout(panel, key, nonlinearity, start, bounds):\\n    u, target = panel[\'u\'], panel[key]\\n\\n    def design(x, pars):\\n        frequency, gain = pars\\n        return _affine(x + gain * nonlinearity(frequency * x))\\n\\n    def residual(pars):\\n        d = design(u, pars)\\n        return (d @ _fit(d, target) - target).ravel()\\n\\n    pars = least_squares(residual, start, bounds=bounds).x\\n    coef = _fit(design(u, pars), target)\\n    return lambda x: design(x, pars) @ coef\\n\\n\\ndef _softsign(x):\\n    return x / (1.0 + np.abs(x))\\n\\n\\ndef _v2_curvature(panel):\\n    # Fit v2 = intercept + (u + curvature(u)) @ mixing.\\n    # Work in the unmixed coordinates for the scalar distance kernel.\\n    u, target = panel[\'u\'], panel[\'v2\']\\n\\n    def residual(params):\\n        k = params[0]\\n        d = _affine(u + _logcosh(k * u) / k)\\n        return (d @ _fit(d, target) - target).ravel()\\n\\n    k = least_squares(residual, (1.1,), bounds=((0.1,), (4.0,))).x[0]\\n    return lambda x: _logcosh(k * x) / k\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    x = np.concatenate((train[\'x\'], test[\'x\']), axis=0)\\n    a = _panel_a(study[\'panel_a\'])\\n    b = _fit(_affine(study[\'panel_b\'][\'u\']), study[\'panel_b\'][\'v\'])\\n    latent = _inverse(x, a)\\n    ib = _affine(latent) @ b\\n    panel = study[\'panel_c\']\\n    v1 = _readout(panel, \'v1\', _softsign, (0.83, -7.3), ((0.05, -30), (15, 0)))\\n    curvature = _v2_curvature(panel)\\n    r = _inverse(latent, b)\\n    i2b = _inverse(r - 0.25 * curvature(r), b)\\n    return {\'z\': {\'x\': x, \'ib\': ib, \'i2b\': i2b, \'i1\': v1(ib)},\\n            \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _standardize(x, n):\\n    return (x - x[:n].mean(axis=0)) / np.maximum(x[:n].std(axis=0), 1e-12)\\n\\n\\ndef _scalar(z, y):\\n    n = len(y)\\n    x = _standardize(z[\'x\'], n)\\n    w = _standardize(z[\'i2b\'], n)\\n    trend = (1 + x @ x.T / x.shape[1]) ** 2\\n    d = ((w[:, None, :] - w[None, :, :]) / 1.5) ** 2\\n    kernel = trend + np.exp(-d.sum(axis=2) / 2) + np.exp(-d / 2).mean(axis=2) ** 2\\n    m = y.mean()\\n    weights = np.linalg.solve(kernel[:n, :n] + 0.008 * np.eye(n), y - m)\\n    return m + kernel[n:, :n] @ weights\\n\\n\\ndef _trajectory(z, targets):\\n    n = len(targets)\\n    x = z[\'ib\']\\n    cols = [x]\\n    cols.extend(x[:, i:i+1] * x[:, j:j+1] for i in range(4) for j in range(i, 4))\\n    cols.extend(f(x) for f in (np.sin, np.cos, np.tanh, np.arctan))\\n    cols.append(x / (1 + x*x))\\n    a = _standardize(np.column_stack(cols), n)\\n    targets_mean = targets.mean(axis=0)\\n    u, s, vh = np.linalg.svd(a[:n], full_matrices=False)\\n    return targets_mean + ((a[n:] @ vh.T) * (s / (s*s + 0.01))) @ (u.T @ (targets - targets_mean))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    z = data[\'z\']\\n    y = _scalar(z, data[\'y\'])\\n    state = _trajectory(z, data[\'traj\'])\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n"}'

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
