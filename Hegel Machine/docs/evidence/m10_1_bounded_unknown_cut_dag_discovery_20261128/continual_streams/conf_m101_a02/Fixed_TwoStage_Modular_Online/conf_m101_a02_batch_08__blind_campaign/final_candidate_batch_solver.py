"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "4c6c369c2c4853e1aebeaf38a2199f7ab481748d42a2efd10f2b387c09863dce", "node_01": "0e8760dcbe1b2084ad108374142e72883056d98c7903f00fc2139708bf41bddf"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover reference coordinates and apply the second panel\'s nonlinear gain.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _a_design(x, j):\\n    a, b = x[:, j], x[:, (j + 1) % 4]\\n    return np.column_stack((np.ones(len(x)), a, b, a*b, a**3))\\n\\n\\ndef _fit_a(u, v):\\n    designs = [_a_design(u, j) for j in range(4)]\\n    weights = np.ones(len(u))\\n    for _ in range(20):\\n        coef = np.array([np.linalg.lstsq(d*weights[:, None], v[:, j]*weights, rcond=None)[0]\\n                         for j, d in enumerate(designs)])\\n        error = np.linalg.norm(np.column_stack([d@c for d, c in zip(designs, coef)]) - v, axis=1)\\n        weights = 1/np.hypot(1, error/max(np.median(error), 1e-8))\\n    keep = np.argsort(error)[:-3]\\n    coef = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]\\n                     for j, d in enumerate(designs)]).T\\n    inverse_seed = np.linalg.lstsq(_affine(v[keep]), u[keep], rcond=None)[0]\\n    return coef, inverse_seed\\n\\n\\ndef _invert_a(v, coef, inverse_seed):\\n    bias, lin, neighbor, pair, cubic = coef\\n    x = _affine(v) @ inverse_seed\\n    j = np.arange(4)\\n    for _ in range(20):\\n        b = np.roll(x, -1, axis=1)\\n        residual = bias + lin*x + neighbor*b + pair*x*b + cubic*x**3 - v\\n        if np.max(np.abs(residual), initial=0) < 1e-11:\\n            break\\n        jac = np.zeros((len(x), 4, 4))\\n        jac[:, j, j] = lin + pair*b + 3*cubic*x*x\\n        jac[:, j, (j+1)%4] = neighbor + pair*x\\n        x -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -0.5, 0.5)\\n    return x\\n\\n\\ndef _b_design(x, k):\\n    return np.column_stack((_affine(x), x*(-np.expm1(-k*x*x))))\\n\\n\\ndef _fit_b(u, v):\\n    def score(k):\\n        q = np.linalg.qr(_b_design(u, k))[0]\\n        err = (v - q @ (q.T @ v))/(1 - (q*q).sum(axis=1, keepdims=True))\\n        return np.mean(err**2)\\n    k = minimize_scalar(score, bounds=(0.15, 2.5), method=\'bounded\').x\\n    return k, np.linalg.lstsq(_b_design(u, k), v, rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    coef, seed = _fit_a(a[\'u\'], a[\'v\'])\\n    k, gain = _fit_b(b[\'u\'], b[\'v\'])\\n    return {out: _b_design(_invert_a(study[src][\'x\'], coef, seed), k) @ gain\\n            for out, src in ((\'z_tr\', \'train\'), (\'z_te\', \'test\'))}\\n", "node_01": "\\"\\"\\"Fit the label and trajectory readouts from the frozen panel representation.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef _poly(x):\\n    i, j = np.triu_indices(x.shape[1])\\n    return np.column_stack((np.ones(len(x)), x, x[:, i] * x[:, j]))\\n\\n\\ndef _reference(study, x):\\n    a = study[\'panel_a\']\\n    u, v = a[\'u\'], a[\'v\']\\n    def design(u, j):\\n        p, q = u[:, j], u[:, (j + 1) % 4]\\n        return np.column_stack((np.ones(len(u)), p, q, p*q, p**3))\\n    ds = [design(u, j) for j in range(4)]\\n    w = np.ones(len(u))\\n    for _ in range(20):\\n        c = np.array([np.linalg.lstsq(d*w[:, None], v[:, j]*w, rcond=None)[0] for j, d in enumerate(ds)])\\n        e = np.linalg.norm(np.column_stack([d@b for d, b in zip(ds, c)]) - v, axis=1)\\n        w = 1/np.hypot(1, e/max(np.median(e), 1e-8))\\n    keep = np.argsort(e)[:-3]\\n    bias, lin, neighbor, pair, cubic = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0] for j, d in enumerate(ds)]).T\\n    affine = lambda x: np.column_stack((np.ones(len(x)), x))\\n    q = affine(x) @ np.linalg.lstsq(affine(v[keep]), u[keep], rcond=None)[0]\\n    j = np.arange(4)\\n    for _ in range(20):\\n        p = np.roll(q, -1, axis=1)\\n        r = bias + lin*q + neighbor*p + pair*q*p + cubic*q**3 - x\\n        if np.max(np.abs(r), initial=0) < 1e-11:\\n            break\\n        J = np.zeros((len(q), 4, 4))\\n        J[:, j, j] = lin + pair*p + 3*cubic*q*q\\n        J[:, j, (j+1)%4] = neighbor + pair*q\\n        q -= np.clip(np.linalg.solve(J, r[..., None])[..., 0], -.5, .5)\\n    return q\\n\\n\\ndef _fit(d, y, n, penalty):\\n    scale = np.maximum(np.linalg.norm(d[:n], axis=0), 1e-10)\\n    d = d / scale\\n    a = d[:n]\\n    return d[n:] @ np.linalg.solve(a.T@a + penalty*np.eye(a.shape[1]), a.T@y)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s, enc = inputs[\'__input__\'], inputs[\'node_00\']\\n    tr = s[\'train\']\\n    n = len(tr[\'x\'])\\n    z = np.concatenate((enc[\'z_tr\'], enc[\'z_te\']))\\n    u = _reference(s, np.concatenate((tr[\'x\'], s[\'test\'][\'x\'])))\\n    a = _poly(z)\\n    trajectory = np.column_stack((a, u, (z[:, :, None]*u[:, None, :]).reshape(len(z), 16)))\\n    target = np.asarray(tr[\'y\']).reshape(-1)\\n    y = np.mean([_label_model(u, z, target, (3, (2,), \'sparse\', 2)), _label_model(u, z, target, (3, (2, 3), \'sparse\', 1))], axis=0)\\n    state = _fit(trajectory, tr[\'traj\'], n, 1e-6)\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n\\ndef _label_features(u, z):\\n    n = len(u)\\n    a = np.column_stack((np.ones(n), z))\\n    w = u * np.roll(z, 1, axis=1)\\n    balance = w.sum(axis=1, keepdims=True)\\n    base = np.column_stack((a, balance))\\n    bases = [base, np.column_stack((a, w)), np.column_stack((base, u)), np.column_stack((base, _poly(z)[:, 5:]))]\\n    mix = (u[:, :, None] * z[:, None, :]).reshape(n, 16)\\n    cross = [mix, w**2, w**3, np.abs(w)]\\n    uni, pairs, aggregate, cubic = [], [], [], []\\n    for q in (u, z, w):\\n        uni.extend([q, q**2, q**3, q**4, np.abs(q), q*np.abs(q)])\\n        for k in (.75, 1.5, 3.):\\n            uni.extend([np.sin(k*q), np.cos(k*q), np.tanh(k*q), q/(1+k*q*q), 1/(1+k*q*q)])\\n        for i in range(4):\\n            for j in range(i+1, 4):\\n                for sign in (-1, 1):\\n                    d = q[:, i] + sign*q[:, j]\\n                    pairs.extend([d**2, d**3, d**4, np.abs(d), np.sin(1.5*d), np.cos(1.5*d), 1/(1+d*d)])\\n        for i in range(4):\\n            for j in range(i, 4):\\n                for k in range(j, 4):\\n                    cubic.append(q[:, i]*q[:, j]*q[:, k])\\n        aggregate.extend([q.sum(1), (q*q).sum(1), (q**3).sum(1), (q**4).sum(1), (q*np.abs(q)).sum(1), np.abs(q).sum(1), np.prod(q, axis=1)])\\n    dirs = np.column_stack((balance, z @ [-1,-1,1,1], z @ [-1,-1,1,1] + .7*balance[:,0], z[:,0]-z[:,1], z[:,2]-z[:,3]))\\n    for k in (.5, 1., 2., 4.):\\n        aggregate.extend([np.sin(k*dirs), np.cos(k*dirs), np.tanh(k*dirs), dirs/(1+k*dirs**2), 1/(1+k*dirs**2)])\\n    aggregate.extend([dirs**2, dirs**3, np.abs(dirs)])\\n    cross.extend([np.column_stack([z[:,i]*np.roll(u,k,axis=1)[:,i]**p for k in range(4) for i in range(4)]) for p in (2,3,4)])\\n    cross.extend([np.column_stack([u[:,i]*np.roll(z,k,axis=1)[:,i]**p for k in range(4) for i in range(4)]) for p in (2,3)])\\n    cross.extend([z*balance, u*balance, w*balance])\\n    groups = [np.column_stack(q) for q in (uni,pairs,aggregate,cross,cubic)]\\n    return bases, groups\\n\\n\\ndef _label_model(u, z, t, spec, fallback=None):\\n    ib, igr, mode, strength = spec\\n    if mode == \'legacy\':\\n        return fallback\\n    bases, groups = _label_features(u, z)\\n    n = len(t)\\n    a = bases[ib]\\n    f = np.column_stack([groups[j] for j in igr]) if igr else np.empty((len(z),0))\\n    q, r = np.linalg.qr(a[:n], mode=\'reduced\')\\n    basecoef = np.linalg.solve(r, q.T@t)\\n    basepred = a[n:]@basecoef\\n    if not f.shape[1]:\\n        return basepred\\n    proj = np.linalg.solve(r, q.T@f[:n])\\n    f = f-a@proj\\n    norm = np.linalg.norm(f[:n], axis=0)\\n    keep = norm > 1e-5\\n    f = f[:,keep]/norm[keep]\\n    g = f[:n]\\n    y = t-q@(q.T@t)\\n    if mode == \'ridge\':\\n        return basepred+f[n:]@g.T@np.linalg.solve(g@g.T+strength*np.eye(n), y)\\n    active = []\\n    pred = basepred\\n    for _ in range(int(strength)):\\n        d = np.column_stack((a[:n], g[:,active]))\\n        q1 = np.linalg.qr(d, mode=\'reduced\')[0]\\n        e = t-q1@(q1.T@t)\\n        gr = g-q1@(q1.T@g)\\n        h = np.maximum(1-(q1*q1).sum(1), 1e-8)\\n        grnorm = np.linalg.norm(gr, axis=0)\\n        v = gr / np.maximum(grnorm, 1e-8)\\n        e_loo = (e[:,None]-v*(v.T@e))/np.maximum(h[:,None]-v*v, 1e-8)\\n        loss = np.mean(np.abs(e_loo), axis=0)\\n        loss[grnorm<1e-5] = np.inf\\n        if np.all(~np.isfinite(loss)) or loss.min() >= .995*np.mean(np.abs(e/h)):\\n            break\\n        active.append(int(loss.argmin()))\\n        ds = np.column_stack((a, f[:,active]))\\n        pred = ds[n:] @ np.linalg.lstsq(ds[:n], t, rcond=None)[0]\\n    return pred\\n"}'

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
