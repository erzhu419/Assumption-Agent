"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "1b7abc5d4e695349feece805fea78e0be97341eba48da4d0bfd57b747f2a007b"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize, least_squares\\nfrom scipy.spatial.distance import cdist\\n\\ndef affine(panel, key, robust=False):\\n    u, v = panel[\'u\'], panel[key]\\n    f = np.column_stack((np.ones(len(u)),u)); ids=np.arange(len(u))\\n    for _ in range(7 if robust else 1):\\n        c=np.linalg.lstsq(f[ids],v[ids],rcond=None)[0]\\n        e=np.linalg.norm(f@c-v,axis=1)\\n        if robust: ids=np.flatnonzero(e<max(.001,4*np.median(e)))\\n    return c\\n\\ndef apply(c,q): return q@c[1:]+c[0]\\n\\ndef temporal(t,z):\\n    def basis(p):\\n        k,d,w=p\\n        return np.column_stack((-np.expm1(-k*t),np.exp(-d*t)*np.sin(w*t)))\\n    def res(p):\\n        f=basis(p);return (f@np.linalg.lstsq(f,z.T,rcond=None)[0]-z.T).ravel()\\n    return np.linalg.qr(basis(least_squares(res,[.5,.2,1.1],bounds=([.01,0,.4],[4,2,2])).x))[0]\\n\\ndef blocks(s,x,n,mode):\\n    a=affine(s[\'panel_a\'],\'v\',True); b=affine(s[\'panel_b\'],\'v\'); c=affine(s[\'panel_c\'],\'v1\')\\n    q=apply(a,x); p=apply(b,q); v=apply(c,q)\\n    def scale(u): return (u-u[:n].mean(0))/np.maximum(u[:n].std(0),1e-6)\\n    def gram(u): return u@u.T\\n    ans=[np.ones((len(x),len(x)))]; names=[\'const\']\\n    def add(k,name):\\n        ans.append(k);names.append(name)\\n    for key,u in [(\'A\',q),(\'raw\',x)]:\\n        g=gram(scale(u))/4\\n        for j in range(1,5 if mode!=\'smooth\' else 3): add(g**j,key+str(j))\\n        d=cdist(scale(u),scale(u),\'sqeuclidean\')\\n        for r in (.8,1.6,3.2,6.4): add(np.exp(-d/(2*r*r)),key+\'rbf\'+str(r))\\n        d=np.sqrt(5*d)/2\\n        add((1+d+d*d/3)*np.exp(-d),key+\'matern\')\\n    if mode in (\'panel\',\'sparse\'):\\n        for key,u in [(\'A\',q),(\'BA\',p),(\'C\',v),(\'raw\',x)]:\\n            g=gram(scale(u))/4\\n            for k in (.6,1.3):\\n                for fn in (np.sin,np.tanh,np.cos):\\n                    f=fn(k*u); f-=f[:n].mean(0); h=gram(f)\\n                    add(h,key+fn.__name__+str(k)); add(h*g,key+fn.__name__+\'cross\'+str(k));add(h*h,key+fn.__name__+\'sq\'+str(k))\\n    if mode==\'sparse\':\\n        for key,u in [(\'A\',q),(\'BA\',p),(\'C\',v)]:\\n            u=scale(u)\\n            for j,f in enumerate(u.T):\\n                for k in (3,4): add(np.outer(f**k,f**k),key+str(j)+\'p\'+str(k))\\n    return np.array(ans),names\\n\\ndef predict(ks,y,prior=.02):\\n    n=len(y); k=ks[:,:n,:n]; l=ks[:,n:,:n]; h=np.maximum(np.diagonal(k,axis1=1,axis2=2).mean(1),1e-12)\\n    I=np.eye(n); k=np.concatenate((k/h[:,None,None],I[None])); l=np.concatenate((l/h[:,None,None],np.zeros_like(l[:1]))); mu=y.mean(0); sd=np.maximum(y.std(0),1e-10); z=(y-mu)/sd\\n    if z.ndim==1:z=z[:,None]\\n    m=z.shape[1]; kf=k.reshape(len(k),-1); lf=l.reshape(len(l),-1)\\n    p0=np.full(len(k),-5.);p0[-1]=-6\\n    def fun(p):\\n        w=np.exp(p); C=cho_factor((np.einsum(\'i,ij->j\',w,kf)).reshape(n,n),check_finite=False)\\n        a=cho_solve(C,z,check_finite=False); inv=cho_solve(C,I,check_finite=False)\\n        return .5*np.sum(a*z)+m*np.log(np.diag(C[0])).sum()+.5*prior*np.sum((p-p0)**2), .5*w*(np.einsum(\'ij,j->i\',kf,(m*inv-a@a.T).ravel()))+prior*(p-p0)\\n    pp=minimize(fun,p0,jac=True,method=\'L-BFGS-B\',bounds=[(-16,4)]*len(k),options={\'maxiter\':180,\'ftol\':1e-9}).x\\n    w=np.exp(pp); C=cho_factor((np.einsum(\'i,ij->j\',w,kf)).reshape(n,n),check_finite=False); a=cho_solve(C,z,check_finite=False)\\n    pred=(np.einsum(\'i,ij->j\',w,lf)).reshape(-1,n)@a*sd+mu\\n    return pred,w\\n\\n\\ndef panel_operator(u, v, quadratic):\\n    def design(x, w):\\n        terms = (x, x*x) if quadratic else (x,)\\n        return np.column_stack((np.ones(len(x)), *terms, np.sin(x*w), np.cos(x*w)))\\n\\n    def residual(w):\\n        f = design(u, w)\\n        return (f @ np.linalg.lstsq(f, v, rcond=None)[0] - v).ravel()\\n\\n    fit = min((least_squares(residual, np.full(u.shape[1], start),\\n                             bounds=(.3, 5), max_nfev=120)\\n               for start in (1., 2., 3.5, 4.5)), key=lambda result: result.cost)\\n    w = fit.x\\n    coef = np.linalg.lstsq(design(u, w), v, rcond=None)[0]\\n    return lambda x: design(x, w) @ coef\\n\\n\\ndef operator_kernels(u, n):\\n    u = (u - u[:n].mean(0)) / np.maximum(u[:n].std(0), 1e-7)\\n    g = u @ u.T / u.shape[1]\\n    return np.array([g, g*g, np.exp(-cdist(u, u, \'sqeuclidean\')/8)])\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    x = np.concatenate((train[\'x\'], study[\'test\'][\'x\']))\\n    n = len(train[\'x\'])\\n    a = affine(study[\'panel_a\'], \'v\', True)\\n    b = affine(study[\'panel_b\'], \'v\')\\n    q = apply(a, x)\\n    r = apply(b, q)\\n    panel = study[\'panel_c\']\\n    c = panel_operator(panel[\'u\'], panel[\'v1\'], False)\\n    d = panel_operator(panel[\'u\'], panel[\'v2\'], True)\\n    extra = np.concatenate([operator_kernels(u, n) for u in (c(q), d(q), c(r), d(r))])\\n    panel_k = np.concatenate((blocks(study, x, n, \'panel\')[0], extra))\\n    smooth_k = blocks(study, x, n, \'smooth\')[0]\\n    target_panel = predict(panel_k, train[\'y\'])[0][:, 0]\\n    target_smooth = predict(smooth_k, train[\'y\'])[0][:, 0]\\n    y_hat = .8*target_panel + .2*target_smooth\\n    basis = temporal(study[\'eval_times\'], train[\'traj\'])\\n    state_k = np.concatenate((smooth_k, extra))\\n    state = predict(state_k, train[\'traj\'] @ basis)[0] @ basis.T\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), state[:, -1].mean(), state[:, -1].std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state, \'diagnostics_vec\': diagnostics}\\n"}'

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
