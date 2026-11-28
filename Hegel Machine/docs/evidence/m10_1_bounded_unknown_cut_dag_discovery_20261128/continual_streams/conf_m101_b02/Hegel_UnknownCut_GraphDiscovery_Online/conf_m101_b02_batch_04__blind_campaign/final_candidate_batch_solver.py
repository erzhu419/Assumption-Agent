"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "2f51626ad83514efebd7e80e742e60b3d05f33b8ac2c1f3e99f1062c23fcdfcc", "node_01": "e3ae7610245774f0f1c1ea93196fde7bd6472479f7858a7ade6d4e16629a1ee8"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _design(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _warp(x, k, c):\\n    return x * (1 + c * (-np.expm1(-(k*x)**2)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s = inputs[\'__input__\']\\n    a, b = s[\'panel_a\'], s[\'panel_b\']\\n    A = _design(a[\'u\'])\\n    v = a[\'v\']\\n    p = np.linalg.lstsq(A, v, rcond=None)[0]\\n    for _ in range(16):\\n        w = 1 / np.sqrt(1 + (np.linalg.norm(A @ p-v, axis=1)/.04)**2)\\n        p = np.linalg.lstsq(A*w[:, None], v*w[:, None], rcond=None)[0]\\n    r = np.linalg.norm(A @ p-v, axis=1)\\n    good = r < max(.01, 8*np.median(r))\\n    if good.sum() >= A.shape[1]:\\n        p = np.linalg.lstsq(A[good], v[good], rcond=None)[0]\\n    x = np.concatenate((s[\'train\'][\'x\'], s[\'test\'][\'x\']))\\n    z = np.linalg.solve(p[1:].T, (x-p[0]).T).T\\n    u, v = b[\'u\'], b[\'v\']\\n    def profile(q):\\n        m = _design(_warp(u, *q))\\n        fit = np.linalg.lstsq(m, v, rcond=None)[0]\\n        return m, fit\\n    def residual(q):\\n        m, fit = profile(q)\\n        return (m @ fit-v).ravel()\\n    q = least_squares(residual, [1., -10.], bounds=([.3, -50.], [2., -.5])).x\\n    m, fit = profile(q)\\n    return {\'z\': z, \'b\': _design(_warp(z, *q)) @ fit,\\n            \'calibration_error\': np.array([np.sqrt(np.mean((A[good] @ p-a[\'v\'][good])**2)), np.sqrt(np.mean((m @ fit-v)**2))])}\\n", "node_01": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.linalg import solve\\nfrom scipy.spatial.distance import cdist\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _design(*blocks):\\n    return np.column_stack((np.ones(len(blocks[0])), *blocks))\\n\\n\\ndef _quadratic(x):\\n    return np.column_stack([x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2)])\\n\\n\\ndef _calibrate(panel, z):\\n    u, v1, v2 = (np.asarray(panel[k]) for k in (\'u\', \'v1\', \'v2\'))\\n    a = _design(_quadratic(u))\\n    p = np.linalg.lstsq(a, v1, rcond=None)[0]\\n    c = _design(_quadratic(z)) @ p\\n    def profile(k):\\n        x = _design(u, np.sqrt(1 + k*u*u))\\n        pinv = np.linalg.pinv(x)\\n        fit = pinv @ v2\\n        resid = x @ fit - v2\\n        leverage = np.einsum(\'ij,ji->i\', x, pinv)\\n        return np.mean((resid.reshape(len(u), -1) / (1-leverage[:, None]))**2), fit, resid\\n    k = minimize_scalar(lambda k: profile(k)[0], bounds=(.1, 3.), method=\'bounded\').x\\n    _, q, _ = profile(k)\\n    d = _design(z, np.sqrt(1 + k*z*z)) @ q\\n    return c, d\\n\\n\\ndef _predict(features, target, length, nugget, polynomial=False):\\n    n = len(target)\\n    z = (features-features[:n].mean(0))/(.1+features[:n].std(0))\\n    g = (z @ z[:n].T) / z.shape[1]\\n    if polynomial:\\n        k = g + (1 + g / length)**3\\n    else:\\n        k = 1 + g + np.exp(-cdist(z, z[:n], \'sqeuclidean\') / (2*length**2*z.shape[1]))\\n    return k[n:] @ solve(k[:n]+nugget*np.eye(n), target, assume_a=\'pos\')\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s, upstream = inputs[\'__input__\'], inputs[\'node_00\']\\n    z, b = upstream[\'z\'], upstream[\'b\']\\n    c, d = _calibrate(s[\'panel_c\'], z)\\n    y = _predict(np.c_[z, b, c, d, _quadratic(z), _quadratic(b)], s[\'train\'][\'y\'], 3.5, 1e-5)\\n    zb = np.c_[z, b]\\n    y = 0.5 * (y + _predict(zb, s[\'train\'][\'y\'], 3.7, .001, polynomial=True))\\n    traj = _predict(zb, s[\'train\'][\'traj\'], 2.4, .001, polynomial=True)\\n    diagnostics = np.array([y.mean(), y.std(), traj[:, -1].mean(), traj[:, -1].std()])\\n    return {\'y_hat\': y, \'state_pred\': traj, \'diagnostics_vec\': diagnostics}\\n"}'

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
