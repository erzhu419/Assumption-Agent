"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "f7eed9075d879b9612ebc7f4cfd854185bb7c08294ced5ec37433363de366a3c", "node_01": "604e8aaf2f4d49e64817f8f32ff027513f63ab9c64f4d35582f0fe694d1483de"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _design(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibrate_a(panel, measured):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    nxt = np.roll(u, -1, axis=1)\\n    coef = np.empty((4, u.shape[1]))\\n    for j in range(u.shape[1]):\\n        x, t = u[:, j], nxt[:, j]\\n        b = np.column_stack((np.ones(len(u)), x, t, x*t + 0.5*x**3))\\n        coef[:, j] = np.linalg.lstsq(b, v[:, j], rcond=None)[0]\\n    for _ in range(16):\\n        residual = v - (coef[0] + coef[1]*u + coef[2]*nxt + coef[3]*(u*nxt + 0.5*u**3))\\n        norms = np.linalg.norm(residual, axis=1)\\n        scale = max(np.median(norms), 2e-4)\\n        w = 1/np.sqrt(1+(norms/scale)**2)\\n        for j in range(u.shape[1]):\\n            x, t = u[:, j], nxt[:, j]\\n            b = np.column_stack((np.ones(len(u)), x, t, x*t + 0.5*x**3))\\n            coef[:, j] = np.linalg.lstsq(b*w[:, None], v[:, j]*w, rcond=None)[0]\\n    x = np.linalg.solve((np.diag(coef[1]) + np.roll(np.diag(coef[2]), 1, axis=1)), (measured-coef[0]).T).T\\n    idx = np.arange(x.shape[1])\\n    for _ in range(24):\\n        t = np.roll(x, -1, axis=1)\\n        r = coef[0] + coef[1]*x + coef[2]*t + coef[3]*(x*t + 0.5*x**3) - measured\\n        if np.max(np.abs(r)) < 1e-11:\\n            break\\n        jac = np.zeros((len(x), x.shape[1], x.shape[1]))\\n        jac[:, idx, idx] = coef[1] + coef[3]*(t + 1.5*x*x)\\n        jac[:, idx, (idx+1)%len(idx)] = coef[2] + coef[3]*x\\n        x -= np.linalg.solve(jac, r[..., None])[..., 0]\\n    return x\\n\\n\\ndef _features(study):\\n    raw = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    x = _calibrate_a(study[\'panel_a\'], raw)\\n    b = study[\'panel_b\']\\n    bv = _design(x) @ np.linalg.lstsq(_design(b[\'u\']), b[\'v\'], rcond=None)[0]\\n    c = study[\'panel_c\']\\n    u = c[\'u\']\\n    v1 = _design(x*np.roll(x, -1, axis=1)) @ np.linalg.lstsq(\\n        _design(u*np.roll(u, -1, axis=1)), c[\'v1\'], rcond=None)[0]\\n\\n    def cosine_fit(freq):\\n        basis = _design(np.cos(freq*u))\\n        coef = np.linalg.lstsq(basis, c[\'v2\'], rcond=None)[0]\\n        return np.sum((basis@coef-c[\'v2\'])**2), coef\\n\\n    freq = minimize_scalar(lambda f: cosine_fit(f)[0], bounds=(1.5, 3.5), method=\'bounded\').x\\n    v2 = _design(np.cos(freq*x)) @ cosine_fit(freq)[1]\\n    return x, bv, v1, v2\\n\\n\\ndef _pair_products(a, b):\\n    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)\\n\\n\\ndef _project(basis, target, penalty):\\n    \\"\\"\\"Fit a standardized ridge projection using only the training rows.\\"\\"\\"\\n    n = len(target)\\n    center = basis[:n].mean(axis=0)\\n    scale = np.maximum(basis[:n].std(axis=0), 1e-12)\\n    a = (basis - center) / scale\\n    tr = a[:n]\\n    coef = np.linalg.solve(tr.T @ tr + np.diag(penalty),\\n                           tr.T @ (target - target.mean(axis=0)))\\n    return a @ coef\\n\\n\\ndef _readout_features(b, c, d, y, traj):\\n    shared = np.column_stack((b, c, d, _pair_products(b, c)))\\n    scalar_basis = np.column_stack((shared, c*d))\\n    sy = _project(scalar_basis, y, np.full(scalar_basis.shape[1], .002))\\n\\n    # The trajectory mixes the diagonal contraction with an off-diagonal\\n    # v1-v2 interaction. Keep the remaining pairs available with shrinkage.\\n    pairs = _pair_products(c, d)\\n    ix = np.arange(c.shape[1])\\n    diag = ix*c.shape[1] + ix\\n    off = np.setdiff1d(np.arange(pairs.shape[1]), np.r_[diag, 1])\\n    primary = np.column_stack((shared, (c*d).sum(axis=1), pairs[:, 1]))\\n    other = np.column_stack((c[:, :-1]*d[:, :-1] - c[:, -1:]*d[:, -1:],\\n                             pairs[:, off]))\\n    basis = np.column_stack((primary, other))\\n    penalty = np.r_[np.full(primary.shape[1], .002), np.full(other.shape[1], .3)]\\n    _, _, modes = np.linalg.svd(traj - traj.mean(axis=0), full_matrices=False)\\n    st = _project(basis, traj @ modes[:2].T, penalty)\\n    h = np.column_stack((sy, st))\\n    tr = h[:len(y)]\\n    return .25 * (h - tr.mean(axis=0)) / np.maximum(tr.std(axis=0), 1e-12)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    _, b, c, d = _features(study)\\n    h = _readout_features(b, c, d, train[\'y\'], train[\'traj\'])\\n    return {\'h\': h, \'n_train\': len(train[\'x\']), \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_01": "import numpy as np\\nfrom itertools import combinations_with_replacement\\n\\n\\ndef _readout_basis(x):\\n    mixed = np.column_stack([x[:, i]*x[:, j] for i, j in\\n                             combinations_with_replacement(range(x.shape[1]), 2)])\\n    return np.column_stack((np.ones(len(x)), x, .4*mixed))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    f = _readout_basis(upstream[\'h\'])\\n    n = upstream[\'n_train\']\\n    a, xt = f[:n], f[n:]\\n    gram = a@a.T\\n    cross = xt@a.T\\n    eye = np.eye(n)\\n    yh = cross@np.linalg.solve(gram+.0003*eye, upstream[\'y\'])\\n    state = cross@np.linalg.solve(gram+.001*eye, upstream[\'traj\'])\\n    end = state[:, -1]\\n    return {\'y_hat\': yh, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}\\n"}'

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
