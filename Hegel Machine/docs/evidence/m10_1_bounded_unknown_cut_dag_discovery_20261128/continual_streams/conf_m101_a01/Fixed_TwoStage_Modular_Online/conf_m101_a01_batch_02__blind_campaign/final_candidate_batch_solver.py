"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "28a3c9396da2d0f58176c640603eabecd6ee0a6e8e09b67dddff27f0b229f417"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _quad(x):\\n    return np.column_stack([_aff(x)] + [x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2)])\\n\\n\\ndef _ols(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _panels(study, x):\\n    a, b, c = (study[k] for k in (\'panel_a\', \'panel_b\', \'panel_c\'))\\n    au, av = _aff(a[\'u\']), a[\'v\']\\n    good = np.ones(len(au), dtype=bool)\\n    for _ in range(8):\\n        w = _ols(au[good], av[good]); r = np.linalg.norm(au @ w - av, axis=1)\\n        good = r < max(3 * np.median(r), .001)\\n    shift, scale = w[0], np.diag(w[1:])\\n    bw = _ols(_aff(b[\'u\']), b[\'v\'])\\n    bf = lambda z: _aff(z) @ bw\\n    bi = lambda z: np.linalg.solve(bw[1:].T, (z-bw[0]).T).T\\n    roots = dict(x=x, a=x*scale+shift, i=(x-shift)/scale, b=bf(x), j=bi(x), ib=bf((x-shift)/scale), ij=bi((x-shift)/scale), bi=(bf(x)-shift)/scale, ji=(bi(x)-shift)/scale)\\n    z = roots.copy()\\n    for key, func, start in [(\'v1\', np.arctan, 2.4), (\'v2\', np.cos, 1.)]:\\n        u, v = c[\'u\'], c[key]\\n        def design(d, p):\\n            return np.column_stack((_aff(d), func(d*p)))\\n        def residual(p):\\n            f = design(u, p)\\n            return (f @ _ols(f, v)-v).ravel()\\n        p = least_squares(residual, np.full(4, start), bounds=(.25, 5)).x\\n        cv = _ols(design(u, p), v)\\n        for k, d in roots.items():\\n            op = design(d, p) @ cv\\n            z[k+key[-1]] = op\\n            z[k+key[-1]+\'b\'] = bf(op)\\n            z[k+key[-1]+\'j\'] = bi(op)\\n    return z\\n\\n\\ndef _ridge(f, target, lam):\\n    n = len(target)\\n    scale = np.maximum(f[:n].std(0), .08); scale[0] = 1.\\n    g = f / scale\\n    a = g[:n]; b = g[n:]\\n    return b @ np.linalg.solve(a.T@a + lam*np.eye(a.shape[1]), a.T@target)\\n\\n\\ndef _scalar_parts(z, y):\\n    x = z[\'x\']; n = len(y); q = _quad(x); result = []\\n    for key, p in [(\'bi1\', 1.), (\'i2j\', 2.3), (\'bi1\', 1.6)]:\\n        d = z[key]\\n        v = np.sin(p*d)/p if p == 1.6 else d/(1+(p*d)**2)\\n        f = np.c_[q,v,(d[:,:,None]*v[:,None,:]).reshape(-1,16)]\\n        result.append(_ridge(f, y, .07))\\n    d = np.c_[z[\'bi1\'], z[\'bi2\']]\\n    result.append(np.arcsinh(_ridge(_quad(np.c_[x,d]), np.sinh(y), .3)))\\n    tar = 1.65*np.arctanh(np.clip(y/1.65,-.999,.999))\\n    a = _aff(x)\\n    r = z[\'x1\'] - a @ _ols(a[:n],z[\'x1\'][:n]); f = np.c_[a,r]\\n    h = np.zeros((4,4)); h[np.triu_indices(4)] = _ols(q[:n],y)[5:]; h=(h+h.T)/2\\n    ev,ve=np.linalg.eigh(h); u=ve[:,-1]*np.sqrt(abs(ev[-1])); v=ve[:,0]*np.sqrt(abs(ev[0]));\\n    def design(p):\\n        a,b=x@p[:4],x@p[4:]\\n        return np.c_[f,a*b]\\n    def res(p):\\n        d=design(p)[:n]\\n        return np.r_[d@_ols(d,tar)-tar,.03*(np.linalg.norm(p.reshape(2,4),axis=1)-1)]\\n    opt=least_squares(res,np.r_[u+v,u-v],max_nfev=90)\\n    d=design(opt.x)\\n    result.append(1.65*np.tanh((d[n:]@_ols(d[:n],tar))/1.65))\\n    return np.array(result)\\n\\n\\ndef _base_solve(study):\\n    train = study[\'train\']\\n    x = np.concatenate((train[\'x\'],study[\'test\'][\'x\']))\\n    z = _panels(study,x)\\n    y = _scalar_parts(z, train[\'y\']).mean(0)\\n    preds=[]\\n    for k in (\'ib2j\',\'b2j\',\'i2j\'):\\n        d=z[k]; f=np.c_[_quad(x),_quad(d)[:,1:]]\\n        preds.append(_ridge(f,train[\'traj\'],.01))\\n    state = np.mean(preds,axis=0)\\n    end = state[:,-1]\\n    return {\'y_hat\':y,\'state_pred\':state,\'diagnostics_vec\':np.array([y.mean(),y.std(),end.mean(),end.std()])}\\n\\n\\ndef _library(study,x,y,tr):\\n    z=_panels(study,x); n=len(y); q=_quad(x)\\n    roots={k:z[k] for k in (\'x\',\'a\',\'i\',\'b\',\'j\',\'ib\',\'ij\',\'bi\',\'ji\')}\\n    cov=np.zeros((4,4)); cov[np.triu_indices(4)]=_ols(q[:n],y)[5:];cov=(cov+cov.T)/2\\n    _,v=np.linalg.eigh(cov);roots[\'e\']=x@v[:,[0,3,1,2]]\\n    _,_,v=np.linalg.svd(tr,full_matrices=False)\\n    amp=tr@v[:2].T\\n    h=q@_ols(q[:n],amp)\\n    yield (\'state\',\'xamp\'),np.c_[q,h*h,h[:,0]*h[:,1],(x[:,:,None]*h[:,None,:]).reshape(-1,8)]\\n    for key,d in roots.items():\\n        if key==\'e\':\\n            yield (key,\'ridges\'),np.c_[q,d[:,:2]**3,d[:,:2]**4,d[:,:2]**5]\\n        cubic=np.column_stack([np.prod(d[:,idx],axis=1) for idx in combinations_with_replacement(range(4),3)])\\n        yield (key,\'cube\'),np.c_[q,cubic]\\n        yield (key,\'diag\'),np.c_[q,d**3,d**4,d**5]\\n        pairs=np.column_stack([d[:,i]+sign*d[:,j] for i in range(4) for j in range(i+1,4) for sign in (-1,1)])\\n        for p in (0.8,1.45,2.15):\\n            for mode in (\'s\',\'t\',\'r\'):\\n                if mode==\'s\': v=np.sin(p*d); w=np.cos(p*d)\\n                elif mode==\'t\': v=np.tanh(p*d);w=v*v\\n                else: v=d/(1+(p*d)**2);w=v*v\\n                yield (key,mode,p,\'sat\'),_quad(v)\\n                yield (key,mode,p,\'mix\'),np.c_[q,v,w,(d[:,:,None]*v[:,None,:]).reshape(-1,16)]\\n                yield (key,mode,p,\'add\'),np.c_[q,v,w]\\n            yield (key,\'pairs\',p),np.c_[q,np.sin(p*pairs),np.cos(p*pairs)]\\n        if key==\'e\':continue\\n        v,w=z[key+\'1\'],z[key+\'2\']\\n        for label,a in [(\'1\',v),(\'2\',w),(\'12\',np.c_[v,w])]:\\n            yield (key,label,\'add\'),np.c_[q,a,a*a]\\n            yield (key,label,\'xprod\'),np.c_[q,a,a*a,(x[:,:,None]*a[:,None,:]).reshape(len(x),-1)]\\n            yield (key,label,\'quad\'),_quad(np.c_[x,a])\\n        for label in (\'1\',\'1j\',\'2\',\'2j\'):\\n            a=z[key+label]\\n            for p in (1.,1.7,2.3):\\n                for mode in (\'s\',\'r\'):\\n                    v=np.sin(p*a)/p if mode==\'s\' else a/(1+(p*a)**2)\\n                    yield (key,label,p,mode),np.c_[q,v,(a[:,:,None]*v[:,None,:]).reshape(-1,16)]\\n\\ndef _fits(f,y):\\n    n=len(y); scale=np.maximum(f[:n].std(0),.05);scale[0]=1\\n    f=f/scale;U,S,V=np.linalg.svd(f[:n],full_matrices=False); sq=S*S; J=f[n:]@V.T\\n    for link in (0,1,1.4,1.75,2.4,-1.,-1.6):\\n        if link>0 and abs(y).max()>=link:continue\\n        t=y if not link else np.arctanh(y/link)*link if link>0 else np.sinh(y/(-link))*(-link)\\n        u=U.T@t\\n        def inv(x):return x if not link else link*np.tanh(x/link) if link>0 else -link*np.arcsinh(x/(-link))\\n        for l in (.006,.03,.15,.55,1.7):\\n            fac=sq/(sq+l); loo=t-(t-U@(fac*u))/np.maximum(1-U**2@fac,1e-8)\\n            yield (link,l),inv(J@(S*u/(sq+l))),np.mean(abs(inv(loo)-y))\\n\\ndef _predict(study, x, y, tr, selected):\\n    s=dict(selected);pred=[]\\n    for k,f in _library(study,x,y,tr):\\n        if k in s:\\n            g={m:p for m,p,_ in _fits(f,y)}\\n            pred.extend(g[m] for m in s[k] if m in g)\\n    return np.mean(pred,axis=0)\\n\\n\\n_SELECTED = [((\'bi\', \'1\', 1.7, \'s\'), [(2.4, 0.15)]), ((\'bi\', \'1\', 1.0, \'r\'), [(2.4, 0.55)]), ((\'i\', \'2j\', 2.3, \'r\'), [(-1.6, 0.15)])]\\n\\ndef solve_batch(study):\\n    out = _base_solve(study)\\n    tr = study[\\"train\\"]\\n    x = np.concatenate((tr[\\"x\\"], study[\\"test\\"][\\"x\\"]))\\n    y = 0.0 * out[\\"y_hat\\"] + 1.0 * _predict(study, x, tr[\\"y\\"], tr[\\"traj\\"], _SELECTED)\\n    out[\\"y_hat\\"] = y\\n    out[\\"diagnostics_vec\\"][:2] = (y.mean(), y.std())\\n    return out\\n"}'

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
