"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "e68e7c28a912d03d931f65f796fc6f45554b88debd1e6bec7babae88e81a11a3"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Panel-calibrated spatial regression and low-rank trajectory prediction.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import least_squares, minimize\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ols(f, y):\\n    return np.linalg.lstsq(f, y, rcond=None)[0]\\n\\n\\ndef _calibrate(panel, x):\\n    f = _aff(panel[\'u\'])\\n    v = panel[\'v\']\\n    keep = np.ones(len(f), dtype=bool)\\n    for _ in range(6):\\n        c = _ols(f[keep], v[keep])\\n        r = np.linalg.norm(f @ c - v, axis=1)\\n        keep = r < max(0.001, 4 * np.median(r))\\n    return _aff(x) @ c\\n\\n\\ndef _degree(x, d):\\n    return np.column_stack([np.prod(x[:, p], axis=1)\\n                            for p in combinations_with_replacement(range(x.shape[1]), d)])\\n\\n\\ndef _cross(x, v):\\n    return (x[:, :, None] * v[:, None, :]).reshape(len(x), -1)\\n\\n\\ndef _gram(f):\\n    return f @ f.T\\n\\n\\ndef _gp(k, y, floor):\\n    n = len(y)\\n    a = k[:, :n, :n]\\n    b = k[:, n:, :n]\\n    scale = np.diagonal(a, axis1=1, axis2=2).mean(axis=1)\\n    eye = np.eye(n)\\n    a = np.concatenate((a / scale[:, None, None], eye[None]))\\n    b = b / scale[:, None, None]\\n    sd = max(float(np.std(y)), 1e-8)\\n    yy = y / sd\\n    flat = a.reshape(len(a), -1)\\n\\n    def objective(p, predict=False):\\n        w = np.exp(p)\\n        c = cho_factor((w @ flat).reshape(n, n), check_finite=False)\\n        alpha = cho_solve(c, yy, check_finite=False)\\n        if predict:\\n            return sd * np.einsum(\'i,ijk,k->j\', w[:-1], b, alpha)\\n        q = cho_solve(c, eye, check_finite=False) - np.outer(alpha, alpha)\\n        return 0.5 * yy @ alpha + np.log(np.diag(c[0])).sum(), 0.5 * w * (flat @ q.ravel())\\n\\n    lower = min(0., np.log(floor / sd**2))\\n    p = minimize(objective, np.r_[np.full(len(k), -2.), -7.], jac=True,\\n                 method=\'L-BFGS-B\', bounds=[(-15, 6)] * len(k) + [(lower, 1)],\\n                 options={\'maxiter\': 180}).x\\n    return objective(p, True)\\n\\n\\ndef _rational(x, f, y):\\n    n = len(y)\\n    squared = x**2\\n\\n    def design(p):\\n        return f / (1. + squared @ p)[:, None]\\n\\n    def residual(p):\\n        a = design(p)[:n]\\n        return a @ _ols(a, y) - y\\n\\n    p = least_squares(residual, np.full(x.shape[1], .2), bounds=(0, 5), max_nfev=40).x\\n    a = design(p)\\n    return a[n:] @ _ols(a[:n], y)\\n\\n\\ndef _features(study, raw):\\n    x = _calibrate(study[\'panel_a\'], raw)\\n    b = study[\'panel_b\']\\n    cb = _ols(_aff(b[\'u\']), b[\'v\'])\\n    c = study[\'panel_c\']\\n    u = c[\'u\']\\n    def diagonal(t):\\n        return np.column_stack((_aff(t), t*t))\\n    cd = _ols(diagonal(u), c[\'v2\'])\\n\\n    def c1_features(t, p):\\n        return _aff(np.tanh(t * p[:4] + p[4:]))\\n\\n    def c1_residual(p):\\n        f = c1_features(u, p)\\n        return (f @ _ols(f, c[\'v1\']) - c[\'v1\']).ravel()\\n\\n    p = least_squares(c1_residual, np.r_[np.full(4, 1.6), np.zeros(4)],\\n                      bounds=(np.r_[np.full(4, .2), np.full(4, -2.)],\\n                              np.r_[np.full(4, 4.), np.full(4, 2.)])).x\\n    cc = _ols(c1_features(u, p), c[\'v1\'])\\n    kernels = {\'base\': _gram(np.column_stack((_aff(x), _degree(x, 2))))}\\n    rational = []\\n    for name, t in [(\'A\', x), (\'B\', _aff(x) @ cb), (\'Br\', _aff(raw) @ cb)]:\\n        q2 = np.column_stack((_aff(t), _degree(t, 2)))\\n        cube = _degree(t, 3)\\n        q3 = np.column_stack((q2, cube))\\n        v1 = c1_features(t, p) @ cc\\n        v2 = diagonal(t) @ cd\\n        cv2 = _cross(t, v2)\\n        kernels[name+\'3\'] = _gram(cube)\\n        kernels[name+\'4\'] = _gram(_degree(t, 4))\\n        kernels[name+\'C1\'] = _gram(np.column_stack((v1, _cross(t, v1), _degree(v1, 2))))\\n        kernels[name+\'C2\'] = _gram(np.column_stack((cv2, _degree(v2, 2))))\\n        kernels[name+\'iso\'] = _gram(q3 / (1. + .2 * np.sum(t*t, axis=1))[:, None])\\n        kernels[name+\'aniso\'] = _gram(q3 / (1. + (t*t) @ [.4, .5, .25, .1])[:, None])\\n        if name != \'A\':\\n            rational.append((t, np.column_stack((q2, cv2))))\\n            h = np.sin(1.6*t)\\n            kernels[name+\'H\'] = _gram(np.column_stack((h, _cross(h, t))))\\n    d = cdist(x, x, \'sqeuclidean\')\\n    for length in (1, 2, 4):\\n        kernels[\'R\'+str(length)] = np.exp(-d/(2*length**2))\\n    return kernels, rational\\n\\n\\ndef _stack(kernels, names):\\n    return np.stack([kernels[k] for k in names.split()])\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    raw = np.vstack((train[\'x\'], study[\'test\'][\'x\']))\\n    k, rational = _features(study, raw)\\n    joint = _stack(k, \'base A3 B3 Br3 A4 B4 Br4 AC1 BC1 BrC1 AC2 BC2 BrC2 Aiso Biso Briso Aaniso Baniso Braniso R1 R2 R4\')\\n    reduced = _stack(k, \'base A3 B3 BC2 BrC2 Biso Briso Baniso Braniso\')\\n    y = np.mean([_gp(joint, train[\'y\'], .00007), _gp(reduced, train[\'y\'], .00007),\\n                 *(_rational(x, f, train[\'y\']) for x, f in rational)], axis=0)\\n\\n    z = train[\'traj\']\\n    g = np.linalg.svd(z, full_matrices=False)[2][0]\\n    g = g / g[-1]\\n    amplitude = z @ g / (g @ g)\\n    branch = _stack(k, \'base BC2 BrC2\')\\n    harmonic = _stack(k, \'base A3 B3 Br3 BC1 BrC1 BH BrH R1 R2 R4\')\\n    a = (0.55 * _gp(joint, amplitude, .000018) + 0.25 * _gp(branch, amplitude, .000018)\\n         + 0.20 * _gp(harmonic, amplitude, .000018))\\n    state = a[:, None] * g\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
