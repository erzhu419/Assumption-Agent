"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "209abdb464c90fcf694784d2460daf529043e2b9c6927de56354afbc50b26976"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize, minimize_scalar\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _profile(tr, times):\\n    def curve(rate):\\n        return -np.expm1(-rate * times)\\n\\n    def error(rate):\\n        f = curve(rate)\\n        return np.mean((tr - np.outer(tr @ f / (f @ f), f)) ** 2)\\n\\n    rate = minimize_scalar(error, bounds=(0.1, 3.0), method=\'bounded\').x\\n    f = curve(rate)\\n    return tr @ f / (f @ f), f\\n\\n\\ndef _gp(x, xq, v, vq, target, kind):\\n    d, dq = _aff(x), _aff(xq)\\n    tdist = (v[:, None, :] - v[None, :, :]) ** 2\\n    qdist = (vq[:, None, :] - v[None, :, :]) ** 2\\n    n, width = v.shape\\n    p = d.shape[1]\\n    eye = np.eye(n)\\n    rhs = np.column_stack((d, target))\\n\\n    def cov(dist, h):\\n        q = dist @ np.exp(h[:-1])\\n        if kind == \'gaussian\':\\n            return np.exp(-q / 2)\\n        if kind == \'rational\':\\n            return 1 / (1 + q / 2)\\n        r = np.sqrt(q)\\n        return (1 + r + q / 3) * np.exp(-r)\\n\\n    def fitted(h):\\n        c = cho_factor(cov(tdist, h) + np.exp(h[-1]) * eye)\\n        f = cho_solve(c, rhs)\\n        a = d.T @ f[:, :p]\\n        beta = np.linalg.solve(a, d.T @ f[:, p:])\\n        alpha = f[:, p:] - f[:, :p] @ beta\\n        return c, a, beta, alpha\\n\\n    def loss(h):\\n        c, a, beta, alpha = fitted(h)\\n        r = target - d @ beta\\n        return ((n - p) * np.linalg.slogdet(r.T @ alpha)[1]\\n                + target.shape[1] * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(a)[1])\\n                + width * np.var(h[:-1]))\\n\\n    h = minimize(loss, np.r_[np.full(width, -.2), -7.5], method=\'L-BFGS-B\',\\n                 bounds=[(-5, 3)] * width + [(-14, -2)], options={\'maxiter\': 85}).x\\n    _, _, beta, alpha = fitted(h)\\n    return dq @ beta + cov(qdist, h) @ alpha\\n\\n\\ndef _joint_prediction(study):\\n    train = study[\'train\']\\n    x, xq = train[\'x\'], study[\'test\'][\'x\']\\n    z, f = _profile(train[\'traj\'], study[\'eval_times\'])\\n    target = np.column_stack((train[\'y\'], z))\\n    panel = study[\'panel_b\']\\n    op = np.linalg.lstsq(_aff(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    inv = np.linalg.inv(op[1:])\\n    v, vq = x @ inv, xq @ inv\\n    pm = _gp(x, xq, v, vq, target, \'matern\')\\n    pr = _gp(x, xq, v, vq, target, \'rational\')\\n    pg = _gp(x, xq, v, vq, target, \'gaussian\')\\n    y, zq = (0.5 * pm + 0.25 * pr + 0.25 * pg).T\\n    state = np.outer(zq, f)\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n\\n\\ndef _coordinates(study, x):\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    keep = np.linalg.norm(a[\'u\']-a[\'v\'],axis=1)<1\\n    aa = np.linalg.lstsq(_aff(a[\'u\'][keep]),a[\'v\'][keep],rcond=None)[0]\\n    bb = np.linalg.lstsq(_aff(b[\'u\']),b[\'v\'],rcond=None)[0]\\n    u = (x-aa[0]) @ np.linalg.inv(aa[1:])\\n    v = (x-bb[0]) @ np.linalg.inv(bb[1:])\\n    return (x, u, v, _aff(u) @ bb, _aff(x) @ bb)\\n\\n\\ndef _geometry(v, warp, mode):\\n    if warp == \'sin\':\\n        v = np.sin(.9*v)\\n    elif warp == \'tanh\':\\n        v = np.tanh(.7*v)\\n    elif warp == \'soft\':\\n        v = v/np.sqrt(1+.6*v*v)\\n    d = (v[:,None,:]-v[None,:,:])**2\\n    if mode == \'pair\':\\n        return np.stack([d[:,:,i]+d[:,:,j] for i in range(4) for j in range(i+1,4)],axis=2)\\n    if mode == \'add\':\\n        return d\\n    return d.sum(axis=2)\\n\\n\\ndef _kernel(d, kind, ell):\\n    q = d / ell**2\\n    if kind == \'sq\':\\n        k = np.exp(-q/2)\\n    elif kind == \'rq\':\\n        k = 1/(1+q/2)\\n    else:\\n        r = np.sqrt(q)\\n        k = (1+r+q/3)*np.exp(-r)\\n    return k if k.ndim==2 else k.mean(axis=2)\\n\\n\\ndef _readout(estimate, link, cap):\\n    if link == \'sqrt\':\\n        return cap-estimate**2\\n    if link == \'log\':\\n        return cap-np.exp(estimate)\\n    return estimate\\n\\n\\ndef _fit_kernel(K, trend, y, i, j, link, cap):\\n    d = trend[i]\\n    if link == \'sqrt\':\\n        y = np.sqrt(cap-y)\\n    elif link == \'log\':\\n        y = np.log(cap-y)\\n    f = cho_solve(cho_factor(K[np.ix_(i,i)]),np.column_stack((d,y)))\\n    u, v = f[:,:-1], f[:,-1]\\n    beta = np.linalg.solve(d.T@u,d.T@v)\\n    return _readout(trend[j]@beta+K[np.ix_(j,i)]@(v-u@beta),link,cap)\\n\\n\\n_BASE_WEIGHT = 0.405\\n_READOUT_MODELS = [((4, \'sin\', \'full\', \'rq\', 2.5, 3e-06, 1, \'id\', 0.0), 0.405), ((3, \'id\', \'full\', \'sq\', 1.6, 3e-06, 1, \'log\', 2.3), 0.09000000000000001), ((2, \'sin\', \'full\', \'sq\', 4.0, 0.0003, 1, \'id\', 0.0), 0.1)]\\n\\n\\ndef solve_batch(study):\\n    result = _joint_prediction(study)\\n    x, xq = study[\'train\'][\'x\'], study[\'test\'][\'x\']\\n    xx = np.vstack((x,xq))\\n    coordinates = _coordinates(study,xx)\\n    d = _aff(xx)\\n    q = np.column_stack([d]+[xx[:,i]*xx[:,j] for i in range(4) for j in range(i,4)])\\n    i, j = np.arange(len(x)), np.arange(len(x),len(xx))\\n    y = _BASE_WEIGHT * result[\'y_hat\']\\n    for (idx,warp,mode,kind,ell,lam,order,link,cap), weight in _READOUT_MODELS:\\n        k = _kernel(_geometry(coordinates[idx],warp,mode),kind,ell)\\n        k.flat[::len(xx)+1] += lam\\n        y += weight*_fit_kernel(k,d if order==1 else q,study[\'train\'][\'y\'],i,j,link,cap)\\n    last = result[\'state_pred\'][:,-1]\\n    result[\'y_hat\'] = y\\n    result[\'diagnostics_vec\'] = np.array([y.mean(),y.std(),last.mean(),last.std()])\\n    return result\\n"}'

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
