"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_23", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__"], "node_02": ["node_00", "node_01"], "node_03": ["node_00", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "81fb3cbcedd636a06052eda88081aabd3520244b16cb85d56b7d08637f86d216", "node_01": "9f2130796f6dec25710f814b513d6ac2bbf79d108106009f418093e2f33a36f9", "node_02": "8191b2fbe0ef047dd00aa1118b60f793228a9027208fb499ffeeeecc165374fb", "node_03": "05c4381ac51f9832b4875fb486f4d8570b53b68e5f7da93cf87dc06c559d4b49"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Fit reference-panel coordinates and learn a compact spatial representation.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\nfrom scipy.special import erf\\nfrom scipy.linalg import solve\\n\\ndef D(x): return np.c_[np.ones(len(x)),x]\\ndef fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]\\ndef cyc(z): return z*np.roll(z,-1,axis=1)\\n\\ndef coordinates(s):\\n    p=s[\'panel_a\']; u,v=p[\'u\'],p[\'v\']; x=np.r_[s[\'train\'][\'x\'],s[\'test\'][\'x\']]\\n    A=D(np.c_[u,cyc(u),u**3]); keep=np.ones(len(u),bool)\\n    for _ in range(12):\\n        w=fit(A[keep],v[keep]); e=np.linalg.norm(A@w-v,axis=1); new=e<max(.001,3*np.median(e))\\n        if new.sum()<18 or np.array_equal(new,keep): break\\n        keep=new\\n    u,v=u[keep],v[keep]\\n    def form(z,q): return D(np.c_[z,np.sin(q[0]*z),z*z])\\n    def err(q):\\n        a=form(u,q); return (a@fit(a,v)-v).ravel()\\n    q=least_squares(err,[2.15],bounds=(1,4)).x; w=fit(form(u,q),v)\\n    z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T\\n    for _ in range(14):\\n        j=w[1:5]+(q[0]*np.cos(q[0]*z))[:,:,None]*w[5:9]+2*z[:,:,None]*w[9:13]\\n        z-=np.clip(np.linalg.solve(j.transpose(0,2,1),(form(z,q)@w-x)[...,None])[...,0],-.3,.3)\\n    p=s[\'panel_b\']; H=lambda a:D(np.sign(a)*np.maximum(abs(a)-.35,0)**1.5); wb=fit(H(p[\'u\']),p[\'v\']); B=lambda a:H(a)@wb\\n    p=s[\'panel_c\']; u=p[\'u\']; wc=fit(D(cyc(u)),p[\'v1\']); C=lambda a:D(cyc(a))@wc\\n    def g(a,q): return D(a+q[1]*a*a/(1+q[0]*a*a))\\n    def e(q):\\n        a=g(u,q); return (a@fit(a,p[\'v2\'])-p[\'v2\']).ravel()\\n    q=least_squares(e,[.25,.45],bounds=([0,-3],[8,3])).x; we=fit(g(u,q),p[\'v2\']); E=lambda a:g(a,q)@we\\n    return z,B,C,E\\n\\ndef features(z,B,C,E):\\n    b=B(z); c=C(z); e=E(z)\\n    basic=[z,b,c,e,B(c),B(e),C(b),E(b),C(e),E(c)]\\n    f=basic.copy()\\n    for x in [z,b,c,e,z+b,z-b,b-.5*z,c+e,c-e,b+c,b-c,b+e,b-e,z+c,z-c,z+e,z-e]:\\n        f.extend([x*x,cyc(x),x*np.roll(x,2,axis=1)])\\n        for a in [.65,1.25,2.15]:\\n            f.extend([np.sin(a*x)-a*x,np.tanh(a*x)-a*x,x/(1+(a*x)**2)-x,np.arctan(a*x)**2])\\n    for i,a in enumerate(basic[:4]):\\n        for b1 in basic[i:4]:\\n            f.append((a[:,:,None]*b1[:,None,:]).reshape(len(z),-1))\\n    return np.column_stack(f)\\n\\ndef head(F,Y):\\n    n=len(Y); A=F[:n]; mean=A.mean(0); scale=np.maximum(A.std(0),.015); X=(F-mean)/scale; A=X[:n]; m=Y.mean(0); y=Y-m\\n    # Sparse Bayesian regression, with shared scale evidence within each response.\\n    result=[]\\n    for k in range(y.shape[1]):\\n        t=y[:,k]; means=[]\\n        for power in [0.,.5,1.]:\\n            a=scale**power; a/=np.median(a); Q=A*a; beta=np.ones(Q.shape[1])/Q.shape[1]; noise=max(.0001*np.var(t),1e-9)\\n            for _ in range(60):\\n                W=Q*beta; inv=solve(W@Q.T+noise*np.eye(n),np.c_[t,Q],assume_a=\'pos\'); w=beta*(Q.T@inv[:,0]); lev=np.sum(Q*inv[:,1:],axis=0); proposed=abs(w)/np.sqrt(np.maximum(lev,1e-10)); beta=np.maximum(1e-11,.35*beta+.65*proposed)\\n            means.append(X@(a*w)+m[k])\\n        result.append(np.mean(means,axis=0))\\n    return np.column_stack(result)\\n\\nfrom scipy.optimize import minimize\\nfrom scipy.linalg import cho_factor,cho_solve\\n\\ndef banks(z,B,C,E):\\n    b,c,e=B(z),C(z),E(z)\\n    q={\'z\':z,\'b\':b,\'c\':c,\'e\':e}\\n    for nm,x in list(q.items()):\\n        if nm!=\'z\':\\n            for name,func in [(\'b\',B),(\'c\',C),(\'e\',E)]:q[name+nm]=func(x)\\n    for nm,x in [(\'z\',z),(\'b\',b),(\'c\',c),(\'e\',e)]:\\n        for nn,y in [(\'z\',z),(\'b\',b),(\'c\',c),(\'e\',e)]:\\n            if nm<nn:\\n                for a in [-1.,1.]:\\n                    x1=x+a*y; k=f\'{nm}{a:+}{nn}\';q[k]=x1\\n                    for name,func in [(\'b\',B),(\'c\',C),(\'e\',E)]:q[name+k]=func(x1)\\n    g={}\\n    for nm,x in q.items():\\n        i,j=np.triu_indices(x.shape[1],1)\\n        g[nm+\'lin\']=x\\n        g[nm+\'quad\']=np.c_[x*x,x[:,i]*x[:,j]]\\n        g[nm+\'cyc\']=cyc(x)\\n        for a in [.65,1.25,2.15]:\\n            for k,h in [(\'tanh\',np.tanh(a*x)),(\'erf\',erf(a*x)),(\'sin\',np.sin(a*x)),(\'rat\',x/(1+(a*x)**2)),(\'th\',np.sign(x)*np.maximum(abs(x)-.22,0)**1.25)]:\\n                g[nm+k+str(a)]=h\\n                if a==1.25:g[nm+k+\'2\']=h*h\\n    for i,(nm,x) in enumerate(list(q.items())[:13]):\\n        for nn,y in list(q.items())[i+1:13]:\\n            g[nm+\'*\'+nn]=(x[:,:,None]*y[:,None,:]).reshape(len(z),-1)\\n    return g\\n\\ndef covs(g,n):\\n    groups=[];names=[]\\n    for name,F in g.items():\\n        F=F-F[:n].mean(0); scale=np.maximum(F[:n].std(0),.008); F=F/np.sqrt(scale); K=F@F.T\\n        K/=max(np.mean(K.diagonal()[:n]),1e-10);groups.append(K);names.append(name)\\n    # Broad smooth components as a fallback.\\n    for name in [\'z\',\'b\',\'c\',\'e\',\'bz\',\'cb\',\'eb\',\'ez\',\'cz\',\'bz-1.0e\']:\\n        if name+\'lin\' not in g:continue\\n        x=g[name+\'lin\']; x=(x-x[:n].mean(0))/np.maximum(x[:n].std(0),.01)\\n        dot=x@x.T/x.shape[1]; dist=np.maximum(dot.diagonal()[:,None]+dot.diagonal()-2*dot,0)\\n        for a in [.08,.3,1.2]:groups.append(np.exp(-a*dist));names.append(name+\'rbf\'+str(a))\\n    return names,np.array(groups)\\n\\ndef regress(K,Y,verbose=False,names=None):\\n    n=len(Y); avg=Y.mean(0); sd=np.maximum(Y.std(0),1e-9); Y=(Y-avg)/sd\\n    flat=np.ascontiguousarray(K[:,:n,:n].reshape(len(K),-1)); cross=np.ascontiguousarray(K[:,n:,:n].reshape(len(K),-1)); eye=np.eye(n); ps=[]\\n    for t in Y.T:\\n        def obj(w):\\n            V=(w[:-1]@flat).reshape(n,n)+(w[-1]+1e-7)*eye\\n            c=cho_factor(V,check_finite=False); a=cho_solve(c,t,check_finite=False); W=cho_solve(c,eye,check_finite=False)-a[:,None]*a[None,:]\\n            return np.log(c[0].diagonal()).sum()+.5*(t@a), np.r_[.5*(flat@W.ravel()),.5*W.trace()]\\n        pp=[]\\n        for penalty in [0.,4.]:\\n            def fn(w):\\n                l,d=obj(w);return l+penalty*np.sum(w[:-1]),d+np.r_[np.full(len(K),penalty),0.]\\n            init=np.r_[np.full(len(K),1./len(K)),.015]; w=minimize(fn,init,jac=True,method=\'L-BFGS-B\',bounds=[(0,15.)]*len(K)+[(.000025,1.)],options={\'maxiter\':145,\'ftol\':4e-8}).x\\n            if verbose and penalty==2.:print(\'groups\',[(names[i],round(w[i],3)) for i in np.argsort(w[:-1])[-12:][::-1] if w[i]>.0001], \'noise\',w[-1],flush=True)\\n            V=(w[:-1]@flat).reshape(n,n)+(w[-1]+1e-7)*eye; cross1=(w[:-1]@cross).reshape(-1,n); a=solve(V,t,assume_a=\'pos\'); pp.append(np.r_[t-w[-1]*a,cross1@a]);\\n        ps.append(np.mean(pp,axis=0))\\n    return np.column_stack(ps)*sd+avg\\n\\ndef extra_groups(g,z,B,C,E):\\n    b,c,e=B(z),C(z),E(z)\\n    vals=[z,b,c,e,b-.5*z,b-z,b+z,b+c,b-c,b+e,b-e,e+c,e-c]\\n    for i,x in enumerate(vals):\\n        opp=np.roll(x,2,axis=1); nxt=np.roll(x,-1,axis=1)\\n        for j,v in enumerate([opp,nxt]):\\n            for s in [-1,1]:\\n                u=x+s*v\\n                for k,a in enumerate([.65,1.25,2.15]):\\n                    h=np.tanh(a*u); r=u/(1+(a*u)**2)\\n                    g[f\'p{i},{j},{s},{k}\']=np.c_[h,r,h*h,r*r]\\n            g[f\'triple{i},{j}\']=x*v*np.roll(x,j+1,axis=1)\\n        for k,a in enumerate([.65,1.25,2.15]):\\n            h=np.arcsinh(a*x); r=x/(1+(a*x)**2)\\n            g[f\'quot{i},{k}\']=np.c_[h,h*h,x*h,r,r*r]\\n    for i,x in enumerate(vals[:4]):\\n        for j,v in enumerate(vals[:4]):\\n            if i!=j:\\n                g[f\'mix{i},{j}\']=np.c_[x/(1+1.25*v*v),x/(1+2.15*v*v),x*np.tanh(1.25*v),x*x*v]\\n    return g\\n\\ndef predict_local(s,variant):\\n    z,B,C,E=coordinates(s); tr=s[\'train\']; t=s[\'eval_times\']; T=tr[\'traj\'].T\\n    def basis(q): return -np.expm1(-t[:,None]*q)\\n    def err(q):\\n        a=basis(q); return (a@fit(a,T)-T).ravel()\\n    a=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)\\n    modes=fit(a,T).T; n=len(modes); rotation=np.linalg.svd(a,full_matrices=False)[2].T\\n    target=np.c_[tr[\'y\'],modes@rotation]\\n    g=banks(z,B,C,E)\\n    if variant==2:g=extra_groups(g,z,B,C,E)\\n    names,K=covs(g,n)\\n    p=regress(K,target)\\n    p[:,1:]=p[:,1:]@rotation.T\\n    return p,modes,z,B(z)\\n\\ndef _baseline(inputs: dict) -> dict:\\n    s=inputs[\'__input__\']; z,B,C,E=coordinates(s); tr=s[\'train\']; t=s[\'eval_times\']; T=tr[\'traj\'].T\\n    def basis(q): return -np.expm1(-t[:,None]*q)\\n    def err(q):\\n        a=basis(q); return (a@fit(a,T)-T).ravel()\\n    a=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)\\n    modes=fit(a,T).T; n=len(modes); F=features(z,B,C,E)\\n    p=head(F,np.c_[tr[\'y\'],modes]); scale=np.maximum(modes.std(0),1e-5)\\n    return {\'y_hat\':p[n:,0].copy(),\'trajectory_design\':np.c_[p[:,1:]/scale,.035*z,.035*B(z)]}\\n\\n_Y_WEIGHTS=[0.0, 1.0, 0.0]\\n_TRAJECTORY=1\\n\\n# Penalized kernel heads with training-only leave-one-out stacking.\\ndef _stack23(z,B,C,E,Y):\\n    from scipy.optimize import minimize\\n    n=len(Y); N=len(z); I=np.eye(n); b,c,e=B(z),C(z),E(z)\\n    mu=Y.mean(0); sd=np.maximum(Y.std(0),1e-8); T=(Y-mu)/sd\\n    coords=[z,b,c,e,np.c_[z,e],np.c_[z,b],np.c_[b,E(b-.5*z)],np.c_[C(b),E(b-z)],np.c_[z,E(b+z)],np.c_[z,b,c,e],np.c_[b,C(b),E(b)],np.c_[z,C(b),E(b)],np.c_[z, B(z+c), C(b+z), E(b+z)]]\\n    gram=[]\\n    for x in coords:\\n        for scheme in (0,1):\\n            X=x-x[:n].mean(0)\\n            if scheme: X=X/np.sqrt(np.maximum(X[:n].std(0),.04))\\n            G=X@X.T; G/=max(G.diagonal()[:n].mean(),1e-8)\\n            d=np.maximum(G.diagonal()[:,None]+G.diagonal()[None,:]-2*G,0)\\n            gram.extend([(1+.35*G)**2,(1+.35*G)**3,(1+.35*G)**4,(1+.35*G)**2+.3*np.exp(-.3*d)])\\n    loo=[]; val=[]\\n    for G in gram:\\n        for ridge in (1e-6,3e-5,.001,.025):\\n            inv=solve(G[:n,:n]+ridge*I,I,assume_a=\'pos\'); a=inv@T\\n            loo.append(T-a/inv.diagonal()[:,None]); val.append(G[:,:n]@a)\\n    L=np.array(loo); V=np.array(val); out=[]\\n    for j in range(T.shape[1]):\\n        U=L[:,:,j].T; t=T[:,j]; errs=np.mean(np.abs(U-t[:,None]),axis=0); ix=np.argsort(errs)[:16]; U=U[:,ix]; m=len(ix)\\n        def objective(w):\\n            err=U@w-t; q=np.sqrt(err*err+.04**2)\\n            return q.mean()+.02*np.dot(w,w), U.T@(err/q)/n+.04*w\\n        res=minimize(objective,np.ones(m)/m,method=\'SLSQP\',jac=True,bounds=[(0,1)]*m,constraints=[{\'type\':\'eq\',\'fun\':lambda w:w.sum()-1.,\'jac\':lambda w:np.ones_like(w)}],options={\'ftol\':1e-9,\'maxiter\':80})\\n        out.append(res.x@V[ix,:,j])\\n    return np.column_stack(out)*sd+mu\\n\\n\\ndef _predict23(s):\\n    z,B,C,E=coordinates(s); tr=s[\'train\']; times=s[\'eval_times\']; T=tr[\'traj\'].T\\n    def basis(q):return -np.expm1(-times[:,None]*q)\\n    def err(q):\\n        f=basis(q);return (f@fit(f,T)-T).ravel()\\n    f=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x); modes=fit(f,T).T\\n    U,sv,R=np.linalg.svd(f,full_matrices=False); rot=R.T; target=np.c_[tr[\'y\'],modes@rot]\\n    _,K=covs(banks(z,B,C,E),len(T.T)); p=regress(K,target); h=_stack23(z,B,C,E,target)\\n    b=B(z); x=np.c_[z,b]; n=len(modes); d=((x[:,None,:]-x[None,:n,:])**2).sum(2); near=np.exp(-d/.20)\\n    return p,h,rot,modes,z,b,near\\n\\ndef _emit23(cache, yw,tw,rw):\\n    p,h,rot,m,z,b,near=cache; n=len(m); M=((1-tw)*p[:,1:]+tw*h[:,1:])@rot.T\\n    if rw:\\n        # Rescale for the frozen kernel\'s dimension-normalized dot product.\\n        q=np.c_[M/np.maximum(m.std(0),1e-5),.035*z,.035*b]\\n        design=np.c_[q*np.sqrt((q.shape[1]+near.shape[1])/q.shape[1]),rw*np.sqrt(q.shape[1]+near.shape[1])*near]\\n    else: design=np.c_[M/np.maximum(m.std(0),1e-5),.035*z,.035*b]\\n    return {\'y_hat\':(1-yw)*p[n:,0]+yw*h[n:,0],\'trajectory_design\':design}\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    return _emit23(_predict23(inputs[\\"__input__\\"]), 1.0, 1.0, 0.02)\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef fit(F, y):\\n    return np.linalg.lstsq(F, y, rcond=1e-10)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    t = study[\'eval_times\']\\n    T = study[\'train\'][\'traj\']\\n\\n    def F(p):\\n        return -np.expm1(-t[:, None] * p)\\n\\n    def err(p):\\n        f = F(p)\\n        return (f @ fit(f, T.T) - T.T).ravel()\\n\\n    f = F(least_squares(err, [1.8, .42], bounds=([1., .15], [3., .85])).x)\\n    return {\'basis\': f, \'modes\': fit(f, T.T).T}\\n", "node_02": "import numpy as np\\n\\n\\ndef kernel(x, degree, decay):\\n    dot = x @ x.T / x.shape[1]\\n    dist = np.maximum(np.diag(dot)[:, None] + np.diag(dot) - 2 * dot, 0)\\n    return (1 + dot)**degree + np.exp(-decay * dist)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    design = inputs[\'node_00\'][\'trajectory_design\']\\n    temporal = inputs[\'node_01\']\\n    modes = temporal[\'modes\']\\n    n = len(modes)\\n\\n    def head(decay):\\n        k = kernel(design, 3, decay)\\n        return k[n:, :n] @ np.linalg.solve(k[:n, :n] + 1e-06 * np.eye(n), modes)\\n\\n    state = (0.25 * head(0.15) + 0.75 * head(1.4)) @ temporal[\'basis\'].T\\n    return {\'state_pred\': state}\\n", "node_03": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    y = inputs[\'node_00\'][\'y_hat\']\\n    state = inputs[\'node_02\'][\'state_pred\']\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state, \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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

def _collect_numeric_leaf_digests(obj) -> set[str]:
    out = set()
    def _walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.add(_canon_digest(v))
        elif isinstance(v, np.ndarray):
            if v.size > 0 and v.dtype.kind in ('f', 'i', 'u'):
                out.add(_canon_digest(v))
        elif isinstance(v, dict):
            for sub in v.values():
                _walk(sub)
        elif isinstance(v, (list, tuple)):
            for sub in v:
                _walk(sub)
    _walk(obj)
    return out

def _collect_subtree_digests(obj) -> set[str]:
    out = {_canon_digest(obj)}
    if isinstance(obj, dict):
        for sub in obj.values():
            out.update(_collect_subtree_digests(sub))
    elif isinstance(obj, (list, tuple)):
        for sub in obj:
            out.update(_collect_subtree_digests(sub))
    return out

class _TrackedPayloadDict(dict):
    def __init__(self, data: dict, src_id: str, consumed_set: set[str]):
        super().__init__(data)
        self._src_id = src_id
        self._consumed_set = consumed_set
    def __getitem__(self, key):
        val = super().__getitem__(key)
        self._consumed_set.add(self._src_id)
        return val
    def get(self, key, default=None):
        if super().__contains__(key):
            self._consumed_set.add(self._src_id)
        return super().get(key, default)
    def values(self):
        if len(self) > 0:
            self._consumed_set.add(self._src_id)
        return super().values()
    def items(self):
        if len(self) > 0:
            self._consumed_set.add(self._src_id)
        return super().items()

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
    raw_input_copy = deepcopy(study)
    raw_input_dig = _canon_digest(raw_input_copy)
    outputs = {'__input__': raw_input_copy}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    edge_consumption_checks = {}
    is_multi_node = len(_SPEC_OBJ['topo_order']) > 1
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        consumed_srcs: set[str] = set()
        if not is_multi_node and srcs == ['__input__']:
            node_in = _TrackedPayloadDict(deepcopy(outputs['__input__']), '__input__', consumed_srcs)
        else:
            node_in = {
                s: (_TrackedPayloadDict(deepcopy(outputs[s]), s, consumed_srcs) if isinstance(outputs[s], dict) else deepcopy(outputs[s]))
                for s in srcs
            }
        in_dig = _canon_digest(outputs['__input__'] if (not is_multi_node and srcs == ['__input__']) else {s: outputs[s] for s in srcs})
        in_num_leaves = set()
        for s in srcs:
            in_num_leaves.update(_collect_numeric_leaf_digests(outputs[s]))
        consumed_srcs.clear()
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        if is_multi_node:
            unused_srcs = sorted(set(srcs) - consumed_srcs)
            if unused_srcs:
                raise RuntimeError(f'DAG node {nid} left declared incoming edges unconsumed: {unused_srcs}')
            edge_consumption_checks[nid] = sorted(consumed_srcs)
        else:
            edge_consumption_checks[nid] = sorted(consumed_srcs) if consumed_srcs else list(srcs)
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Strict anti-wrapper-shell & genuine numeric transformation check (`U06`):
            out_subtrees = _collect_subtree_digests(out_v)
            echoed = (out_dig == in_dig) or (raw_input_dig in out_subtrees) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            out_num_leaves = _collect_numeric_leaf_digests(out_v)
            novel_num_leaves = out_num_leaves - in_num_leaves
            if echoed or not out_num_leaves or not novel_num_leaves:
                raise RuntimeError(f'DAG node {nid} failed anti-wrapper-shell / novel numeric transformation check')
            non_identity_checks[nid] = True
        # Convert any tracked dict views returned by user code into plain dicts before freezing in outputs
        outputs[nid] = {k: (dict(v) if isinstance(v, _TrackedPayloadDict) else v) for k, v in out_v.items()}
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
            'edge_consumption_checks': edge_consumption_checks,
            'all_declared_edges_consumed': all(set(edge_consumption_checks.get(n, [])) == set(_SPEC_OBJ['incoming'][n]) for n in _SPEC_OBJ['topo_order']) if is_multi_node else True,
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
