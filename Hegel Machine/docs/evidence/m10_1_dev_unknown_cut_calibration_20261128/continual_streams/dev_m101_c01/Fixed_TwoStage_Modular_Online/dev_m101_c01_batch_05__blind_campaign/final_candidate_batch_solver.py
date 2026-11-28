"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "145363bbd9160317a3a4106bec6cc62fa538125f924ab7c6e9e24147d0e35e22"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Panel-informed kernel regression for scalar and trajectory observations.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import least_squares, minimize\\n\\n\\ndef _lstsq(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _operators(study):\\n    a, b, c = (study[k] for k in (\'panel_a\', \'panel_b\', \'panel_c\'))\\n    u, v = a[\'u\'], a[\'v\']\\n    A = np.c_[np.ones(len(u)), u]\\n    idx = np.arange(len(u))\\n    for _ in range(7):\\n        p = _lstsq(A[idx], v[idx])\\n        err = np.linalg.norm(A @ p - v, axis=1)\\n        idx = np.argsort(err)[:3 * len(u) // 4]\\n    idx = err <= 5 * np.median(err)\\n    shift, gain = np.array([_lstsq(A[idx][:, [0, j + 1]], v[idx, j]) for j in range(4)]).T\\n    u = b[\'u\']\\n    B = _lstsq(np.c_[np.ones(len(u)), u], b[\'v\'])\\n    u = c[\'u\']\\n    def fit_operator(f, start, bounds, target):\\n        def design(w, p):\\n            return np.c_[np.ones(len(w)), w, f(w, p)]\\n        def res(p):\\n            M = design(u, p)\\n            return (M @ _lstsq(M, target) - target).ravel()\\n        p = least_squares(res, start, bounds=bounds).x\\n        coeff = _lstsq(design(u, p), target)\\n        return lambda w: design(w, p) @ coeff\\n    f1 = fit_operator(lambda w,p: np.tanh(p[0]*w)+p[1]*np.arctan(p[2]*w),\\n                      [1.5,.2,5.7], ([.2,0.,.5],[5.,2.,9.]), c[\'v1\'])\\n    f2 = fit_operator(lambda w,p: np.logaddexp(p[0]*w,-p[0]*w),\\n                      [1.1], ([.2],[4.]), c[\'v2\'])\\n    return shift, gain, B, f1, f2\\n\\n\\ndef _kernels(study, x, n):\\n    shift, gain, B, f1, f2 = _operators(study)\\n    a = (x - shift) / gain\\n    b = B[0] + x @ B[1:]\\n    ab = B[0] + a @ B[1:]\\n    bi = np.linalg.solve(B[1:].T, (x - B[0]).T).T\\n    def normalize(w):\\n        return (w-w[:n].mean(0))/np.maximum(w[:n].std(0), 1e-8)\\n    def poly(w, degree=4):\\n        g = normalize(w)\\n        Q = g @ g.T / g.shape[1]\\n        return [Q**k for k in range(1, degree+1)]\\n    def panel(w):\\n        return poly(f1(w), 3) + poly(f2(w), 3)\\n    def interact(w):\\n        return poly(f1(w), 2) + poly(f2(w), 2) + poly(np.c_[w,f1(w),f2(w)], 3)[1:]\\n    def add(w, pairs=False):\\n        g = normalize(w)\\n        R = np.exp(-.5*(g[:,None]-g[None,:])**2).transpose(2,0,1)\\n        return list(R) + ([R[i]*R[j] for i in range(4) for j in range(i+1,4)] if pairs else [])\\n    def trig(w):\\n        return [poly(f(w), 1)[0] for f in (lambda q:np.arctan(2.3*q),lambda q:np.cos(1.28*q),lambda q:np.sin(1.7*q))]\\n    def radial(w):\\n        g = normalize(w)\\n        q = g @ g.T / g.shape[1]\\n        d = ((g[:, None] - g[None, :])**2).sum(-1)\\n        return [q, q**2] + [np.exp(-h*d) for h in (.15, .5, 1.5)]\\n    def warp(w):\\n        u, v = f1(w), f2(w)\\n        return radial(u) + radial(v) + poly(np.c_[w,u,v])[1:]\\n    def products(w):\\n        q = poly(w, 1)[0]\\n        k = [poly(f(w), 1)[0] for f in (\\n            lambda w: np.tanh(1.7*w),\\n            lambda w: np.logaddexp(1.1*w,-1.1*w),\\n            lambda w: np.arctan(2.3*w),\\n        )]\\n        return k + [q*h for h in k] + [q**2*h for h in k]\\n    raw = poly(x)\\n    groups = {\\n        \'a_panel\': panel(a), \'abi_panel\': panel(np.linalg.solve(B[1:].T,(a-B[0]).T).T),\\n        \'a_interact\': interact(a), \'raw_add\': add(x, True), \'ab_add\': add(ab, True),\\n        \'bi_add\': add(bi, True), \'ab_trig\': trig(ab) + add(ab, True),\\n        \'joint\': panel(a)+add(x)+add(ab)+add(bi), \'plain\': [],\\n    }\\n    constant = np.ones((len(x),len(x)))\\n    kernels = {key: np.array([constant] + raw + more) for key,more in groups.items()}\\n    base = [constant] + raw[:3]\\n    abi = np.linalg.solve(B[1:].T, (a-B[0]).T).T\\n    kernels[\'abi_warp\'] = np.array(base + warp(abi))\\n    kernels[\'ab_product\'] = np.array(base + products(ab))\\n    return kernels\\n\\n\\ndef _predict(kernels, targets):\\n    n, m = targets.shape\\n    K = np.concatenate((kernels[:, :n,:n], np.eye(n)[None]))\\n    E = K[-1]\\n    def objective(p):\\n        weights = np.exp(p)\\n        C = cho_factor(np.einsum(\'i,ijk->jk\',weights,K),lower=True,check_finite=False)\\n        A = cho_solve(C, targets, check_finite=False)\\n        G = m*cho_solve(C,E,check_finite=False)-A@A.T\\n        loss = .5*np.sum(targets*A)+m*np.log(C[0].diagonal()).sum()\\n        return loss, .5*weights*np.einsum(\'jk,ijk->i\',G,K)\\n    init = np.r_[0., np.full(len(K)-2,-3.), -8.]\\n    p = minimize(objective,init,jac=True,method=\'L-BFGS-B\',bounds=[(-16.,5.)]*(len(K)-1)+[(-16.,0.)],options={\'maxiter\':160}).x\\n    weights = np.exp(p)\\n    C = cho_factor(np.einsum(\'i,ijk->jk\',weights,K),lower=True,check_finite=False)\\n    return np.einsum(\'i,ijk->jk\',weights[:-1],kernels[:,n:,:n]) @ cho_solve(C,targets,check_finite=False)\\n\\n\\ndef _predictions(study, scalar_keys, state_keys):\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    K = _kernels(study, np.r_[train[\'x\'],study[\'test\'][\'x\']], n)\\n    y = np.asarray(train[\'y\'])\\n    z = np.asarray(train[\'traj\'])\\n    ym, ys = y.mean(), y.std()\\n    zm, zs = z.mean(0), z.std(0)\\n    zstd = (z-zm)/zs\\n    V = np.linalg.svd(zstd,full_matrices=False)[2][:2]\\n    T = zstd @ V.T\\n    ts = T.std(0)\\n    scalar = {k: _predict(K[k],((y-ym)/ys)[:,None])[:,0]*ys+ym for k in scalar_keys}\\n    state = {k: (_predict(K[k],T/ts)*ts)@V*zs+zm for k in state_keys}\\n    return scalar,state\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    scalar, state = _predictions(study, (\'a_panel\',\'raw_add\',\'abi_panel\',\'abi_warp\'), (\'a_interact\',\'ab_trig\',\'bi_add\',\'ab_product\'))\\n    y = .5*scalar[\'a_panel\']+.25*scalar[\'raw_add\']+.25*scalar[\'abi_panel\']\\n    z = .4*state[\'a_interact\']+.35*state[\'ab_trig\']+.25*state[\'bi_add\']\\n    y = .5*(y+scalar[\'abi_warp\'])\\n    z = .5*(z+state[\'ab_product\'])\\n    end = z[:,-1]\\n    return {\'y_hat\':y,\'state_pred\':z,\'diagnostics_vec\':np.array([y.mean(),y.std(),end.mean(),end.std()])}\\n"}'

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
