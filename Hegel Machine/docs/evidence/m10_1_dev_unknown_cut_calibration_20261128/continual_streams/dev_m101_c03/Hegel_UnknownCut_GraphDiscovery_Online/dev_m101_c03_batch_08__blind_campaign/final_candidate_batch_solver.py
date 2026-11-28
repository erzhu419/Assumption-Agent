"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "284a473e77754b97fca5db0c2de24c04e94338bce3a1f2d4457407c948151760"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated scalar and three-mode trajectory predictors.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _lstsq(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _ones(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _next(x):\\n    return np.roll(x, -1, axis=1)\\n\\n\\ndef _similarity(u, v):\\n    q = u - u.mean(0)\\n    l, s, r = np.linalg.svd(q.T @ (v-v.mean(0)))\\n    m = (l @ r) * (s.sum()/np.sum(q*q))\\n    return m, v.mean(0) - u.mean(0) @ m\\n\\n\\ndef _features(study, x):\\n    p = study[\'panel_a\']\\n    u, v = p[\'u\'], p[\'v\']\\n    a = []\\n    for j in range(u.shape[1]):\\n        q, w = u[:, j], u[:, (j+1) % u.shape[1]]\\n        basis = _ones(np.column_stack((q, w, q*w, q**3)))\\n        y = v[:, j]\\n        weights = np.ones(len(u))\\n        for _ in range(45):\\n            coef = _lstsq(basis*weights[:, None], y*weights)\\n            error = abs(basis @ coef - y)\\n            weights = 1/np.sqrt(np.maximum(error, 1e-4))\\n        keep = error < 0.001\\n        a.append(_lstsq(basis[keep], y[keep]))\\n    bias, diag, cyc, prod, cubic = np.array(a).T\\n\\n    def invert(row):\\n        def residual(z):\\n            w = np.roll(z, -1)\\n            return bias + diag*z + cyc*w + prod*z*w + cubic*z**3 - row\\n        return least_squares(residual, row, gtol=1e-12).x\\n\\n    z = np.array([invert(row) for row in x])\\n    p = study[\'panel_c\']\\n    u = p[\'u\']\\n    m, k = _similarity(u*_next(u), p[\'v1\'])\\n    c = (z*_next(z)) @ m + k\\n\\n    def cosine_fit(rate):\\n        q = np.cos(rate*u)\\n        m, k = _similarity(q, p[\'v2\'])\\n        return np.mean((q @ m + k-p[\'v2\'])**2)\\n\\n    rate = minimize_scalar(cosine_fit, bounds=(1.8, 3), method=\'bounded\').x\\n    m, k = _similarity(np.cos(rate*u), p[\'v2\'])\\n    d = np.cos(rate*z) @ m + k\\n    return z, c, d\\n\\n\\ndef _powers(x, degree):\\n    return np.column_stack([np.prod(x[:, j], axis=1)\\n                            for j in combinations_with_replacement(range(x.shape[1]), degree)])\\n\\n\\ndef _temporal(t, h, groups, pen=0.001, inspect=False):\\n    n, nt = h.shape\\n    groups = [(g-g[:n].mean(0))/np.maximum(g[:n].std(0), 1e-8) for g in groups]\\n    mu = h.mean(0)\\n    reg = np.diag(np.concatenate([np.full(g.shape[1], np.sqrt(pen)*(0.4 if j==1 else 1))\\n                                  for j, g in enumerate(groups)]))\\n    target = np.r_[(h-mu).ravel(), np.zeros(len(reg))]\\n\\n    def design(rate, train):\\n        a, b, c = rate\\n        modes = np.array([-np.expm1(-a*t), t**b, t/(1+c*t)])\\n        modes /= modes[:, -1:]\\n        blocks = []\\n        for mode, g in zip(modes, groups):\\n            g = g[:n] if train else g[n:]\\n            blocks.append((g[:, None, :]*mode[None, :, None]).reshape(len(g)*nt, g.shape[1]))\\n        return np.column_stack(blocks)\\n\\n    def profile(rate):\\n        mat = np.vstack((design(rate, True), reg))\\n        coef = _lstsq(mat, target)\\n        return mat @ coef-target, coef\\n\\n    r = least_squares(lambda rate: profile(rate)[0], [1.8, .5, .4],\\n                      bounds=([.5, .25, .015], [4, .8, 4]), max_nfev=40)\\n    err, coef = profile(r.x)\\n    pred = (design(r.x, False) @ coef).reshape(-1, nt) + mu\\n    return (pred, r.x, coef, np.std(err[:n*nt])) if inspect else pred\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    z, c, d = _features(study, np.vstack((train[\'x\'], study[\'test\'][\'x\'])))\\n    dot = (c*d).sum(1, keepdims=True)\\n    p3 = _powers(z, 3)\\n    scalar = _ones(np.column_stack((c, d, dot, p3, z)))\\n    y = scalar[n:] @ _lstsq(scalar[:n], train[\'y\'])\\n    slow = np.c_[d, c, c[:, 0]*d[:, 1], z]\\n    state = _temporal(study[\'eval_times\'], train[\'traj\'], [np.c_[c, dot], p3, slow])\\n    terminal = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}\\n"}'

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
