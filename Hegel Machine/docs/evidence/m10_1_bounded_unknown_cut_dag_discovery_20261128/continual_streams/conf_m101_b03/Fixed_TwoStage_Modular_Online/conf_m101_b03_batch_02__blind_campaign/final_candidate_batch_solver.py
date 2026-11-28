"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "ae20945c497587f13bc0a43268af81275bb387e8955132c6960846c61ade8582"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated panel interaction model with two relaxation modes.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-9)[0]\\n\\n\\ndef _affine(x):\\n    return np.c_[np.ones(len(x)), x]\\n\\n\\ndef _cyclic(x):\\n    return x * np.roll(x, -1, axis=1)\\n\\n\\ndef _coordinates(study):\\n    panel = study[\'panel_a\']\\n    a, v = _affine(panel[\'u\']), panel[\'v\']\\n    m = _fit(a, v)\\n    for _ in range(12):\\n        w = np.minimum(1., .07 / np.maximum(np.linalg.norm(a @ m - v, axis=1), 1e-12))\\n        m = _fit(a * w[:, None], v * w[:, None])\\n    good = np.linalg.norm(a @ m - v, axis=1) < .01\\n    m = _fit(a[good], v[good])\\n    x = np.vstack([study[\'train\'][\'x\'], study[\'test\'][\'x\']])\\n    return np.linalg.solve(m[1:].T, (x - m[0]).T).T\\n\\n\\ndef _views(study, z):\\n    c = study[\'panel_c\']\\n    u = c[\'u\']\\n    p = _affine(_cyclic(z)) @ _fit(_affine(_cyclic(u)), c[\'v1\'])\\n    def f(x, k):\\n        return np.c_[_affine(x), np.log1p(k * x * x)]\\n    def loss(k):\\n        g = f(u, k)\\n        return np.mean((g @ _fit(g, c[\'v2\']) - c[\'v2\'])**2)\\n    k = minimize_scalar(loss, bounds=(.05, 1.5), method=\'bounded\').x\\n    q = f(z, k) @ _fit(f(u, k), c[\'v2\'])\\n    return p, q\\n\\n\\ndef _temporal(t, s):\\n    def basis(r):\\n        return -np.expm1(-t[:, None] * r)\\n    def residual(r):\\n        h = basis(r)\\n        return (h @ _fit(h, s.T) - s.T).ravel()\\n    h = basis(least_squares(residual, [.4, 2.], bounds=(.001, 12.)).x)\\n    return h, _fit(h, s.T).T\\n\\n\\ndef solve_batch(study):\\n    tr = study[\'train\']\\n    n = len(tr[\'x\'])\\n    z = _coordinates(study)\\n    h, modes = _temporal(study[\'eval_times\'], tr[\'traj\'])\\n    base = np.c_[_affine(z), _cyclic(z), z*z]\\n    read = _fit(np.c_[base[:n], modes], tr[\'y\'])\\n    p, q = _views(study, z)\\n    shared = np.c_[base, q]\\n    slow = np.c_[shared, p[:, 0] * q[:, 1]]\\n    fast = np.c_[shared, np.sum(p*q, axis=1)]\\n\\n    # Estimate both spatial modes jointly from the measured trajectories and response.\\n    gs, gf = slow[:n], fast[:n]\\n    nt = h.shape[0]\\n    temporal = np.concatenate([\\n        gs[:, None, :] * h[None, :, 0, None],\\n        gf[:, None, :] * h[None, :, 1, None],\\n    ], axis=2).reshape(n * nt, gs.shape[1] + gf.shape[1])\\n    design = np.r_[temporal, np.c_[read[-2] * gs, read[-1] * gf]]\\n    target = np.r_[tr[\'traj\'].ravel(), tr[\'y\'] - base[:n] @ read[:-2]]\\n    penalty = .005 * np.linalg.norm(design, axis=0)\\n    w = _fit(np.r_[design, np.diag(penalty)], np.r_[target, np.zeros(len(penalty))])\\n    d = gs.shape[1]\\n    pred = np.c_[slow[n:] @ w[:d], fast[n:] @ w[d:]]\\n    y = base[n:] @ read[:-2] + pred @ read[-2:]\\n    state = pred @ h.T\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}\\n"}'

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
