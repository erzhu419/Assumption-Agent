"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "334bad31a0064ee2fc61e99a864f3590560a0168da12047bbc1cff8fd470d1cf"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated nonlinear predictor for the batch readout and state trajectory.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import solve\\n\\ndef _calibrate(panel, x):\\n    u,v=(np.asarray(panel[k],float) for k in (\'u\',\'v\'))\\n    def features(a,b): return np.column_stack((np.ones(len(a)),a,b,a*b,a**3))\\n    g=np.ones(len(u),bool)\\n    for _ in range(4):\\n        m=np.array([np.linalg.lstsq(features(u[g,j],u[g,(j+1)%4]),v[g,j],rcond=None)[0] for j in range(4)])\\n        err=np.linalg.norm(np.column_stack([features(u[:,j],u[:,(j+1)%4])@m[j] for j in range(4)])-v,axis=1)\\n        g=err<=np.partition(err,44)[44]\\n    z=np.array(x,float,copy=True); j=np.arange(4)\\n    for _ in range(14):\\n        w=np.roll(z,-1,axis=1); J=np.zeros((len(z),4,4)); J[:,j,j]=m[:,1]+m[:,3]*w+3*m[:,4]*z*z; J[:,j,(j+1)%4]=m[:,2]+m[:,3]*z\\n        f=m[:,0]+m[:,1]*z+m[:,2]*w+m[:,3]*z*w+m[:,4]*z**3\\n        z-=np.linalg.solve(J,(f-x)[...,None])[...,0]\\n    return z\\n\\nfrom scipy.optimize import least_squares\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.special import erf\\n\\ndef _panel_features(panel):\\n    u,v=(np.asarray(panel[k],float) for k in (\'u\',\'v\'))\\n    forms=(lambda a,q:np.tanh(q*a)**3,\\n           lambda a,q:a*np.tanh(q*a)**2,\\n           lambda a,q:(a/np.sqrt(1+q*a*a))**3,\\n           lambda a,q:a**3/(1+q*a*a),\\n           lambda a,q:a**3/(1+q*abs(a)),\\n           lambda a,q:np.sin(q*a)**3,\\n           lambda a,q:erf(q*a)**3,\\n           lambda a,q:a**3/(1+q*a**4),\\n           lambda a,q:np.sign(a)*abs(a)**q)\\n    candidates=[]\\n    for f in forms:\\n        for linear in (False,True):\\n            def design(a,q):\\n                g=f(a,q)\\n                return np.c_[np.ones(len(a)),a,g] if linear else np.c_[np.ones(len(a)),g]\\n            def residual(p):\\n                H=design(u,p[0]); return (H@np.linalg.lstsq(H,v,rcond=1e-9)[0]-v).ravel()\\n            for start in (.45,1.3,2.6):\\n                q=least_squares(residual,[start],bounds=(.03,4.),max_nfev=55).x[0]\\n                H=design(u,q); pin=np.linalg.pinv(H,rcond=1e-9); B=pin@v\\n                loo=((H@B-v)/np.maximum(.045,1-np.sum(H*pin.T,axis=1))[:,None]); loss=np.mean(loo**2)\\n                candidates.append((loss,f,q,B,linear))\\n    loss,f,q,B,linear=min(candidates,key=lambda c:c[0]); return f,q,B,linear\\n\\ndef _families(x,panel):\\n    n=len(x); one=np.ones((n,1)); i,j=np.triu_indices(4); k,l=np.triu_indices(4,1)\\n    f,q,B,lin=_panel_features(panel)\\n    g=f(x,q); h=(np.c_[one,x,g] if lin else np.c_[one,g])@B\\n    specs=[(.85,lambda a:a/np.sqrt(1+a*a)),(1.,lambda a:a/np.sqrt(1+a*a)),(1.25,lambda a:a/np.sqrt(1+a*a)),\\n           (1.3,np.arctan),(1.5,np.arctan),(1.7,np.arctan),(1.25,np.tanh),(1.5,np.tanh),(1.65,np.tanh),(1.85,np.tanh),\\n           (1.25,erf),(1.5,erf),(1.6,lambda a:np.clip(a,-1,1.)),(1.,lambda a:a/(1+abs(a)))]\\n    def blocks(z):\\n        return z[:,i]*z[:,j], (x[:,:,None]*z[:,None,:]).reshape(n,16)\\n    for index,(q,f) in enumerate(specs):\\n        z=f(q*x); zz,xz=blocks(z); m=np.c_[one,x,z]; r=np.c_[m,zz,xz]; delta=x-z/(q if index not in (12,13) else max(q,1));\\n        yield 0,index,r\\n        yield 1,index,np.c_[m,zz,.35*xz,x*z]\\n        yield 2,index,np.c_[m,zz,.7*xz,.45*z**3,.35*delta**2,.25*(z[:,k]*z[:,l])*(x[:,k]+x[:,l])]\\n        yield 3,index,np.c_[m,zz,x*z,.4*f(q*(x[:,k]+x[:,l])),.4*f(q*(x[:,k]-x[:,l]))]\\n        yield 4,index,np.c_[r, .6*g, .45*(z[:,:,None]*g[:,None,:]).reshape(n,16)]\\n        yield 5,index,np.c_[r,.65*h,.5*h[:,i]*h[:,j],.4*np.tanh(h),.3*(z[:,:,None]*h[:,None,:]).reshape(n,16)]\\n        for cut in (1.0,1.25,1.5):\\n            c=np.clip(x,-cut,cut); w=f(q*c); ww,wx=blocks(w)\\n            yield 6,index,np.c_[one,c,w,ww,(c[:,:,None]*w[:,None,:]).reshape(n,16)]\\n    z=np.tanh(1.55*x); zz,xz=blocks(z)\\n    for power in (1,2,3):\\n        z=np.sign(g)*abs(g)**(1/power); zz,xz=blocks(z)\\n        yield 7,power,np.c_[one,x,z,zz,xz]\\n\\ndef _portfolio(study,x):\\n    train=study[\'train\']; y=np.asarray(train[\'y\'],float); t=np.asarray(train[\'traj\'],float); n=len(y)\\n    targets=np.c_[y,t]\\n    groups={}\\n    for kind,idx,F in _families(x,study[\'panel_b\']):\\n        K=F@F[:n].T; G=K[:n]; diagonal=np.eye(n)\\n        for reg in (.0003,.0015,.004,.012,.04):\\n            C=cho_factor(G+reg*diagonal); a=cho_solve(C,targets); lev=np.diag(cho_solve(C,diagonal)); E=a/lev[:,None]\\n            oof=targets-E; pred=K[n:]@a\\n            # Pool multiple nearby feature maps rather than selecting a single unstable extrapolant.\\n            loss=np.mean(abs(E[:,0])); tail=np.mean(abs(E[:,0])[np.argsort(np.linalg.norm(x[:n],axis=1))[-max(n//4,1):]])\\n            score=.82*loss+.18*tail\\n            groups.setdefault(kind,[]).append((score,loss,np.mean(abs(E[:,1:])),pred,oof))\\n    out={}; loo={}\\n    for kind,items in groups.items():\\n        items.sort(key=lambda a:a[0]); subset=items[:min(8,len(items))]; w=np.exp(-(np.array([a[0] for a in subset])-subset[0][0])/.009);w/=w.sum()\\n        out[kind]=np.einsum(\'i,ijk->jk\',w,np.array([a[3] for a in subset])); loo[kind]=np.einsum(\'i,ijk->jk\',w,np.array([a[4] for a in subset]));\\n    # Calibrate a low-rank correction between independently estimated output channels.\\n    state=[]\\n    for idx,(q,f) in enumerate([(1.03,lambda u:u/np.sqrt(1+u*u)),(1.15,lambda u:u/np.sqrt(1+u*u)),(1.25,np.arctan),(1.32,np.tanh)]):\\n        z=f(q*x); i,j=np.triu_indices(4); F=np.c_[np.ones(len(x)),x,z,z[:,i]*z[:,j],(x[:,:,None]*z[:,None,:]).reshape(len(x),16)]; K=F@F[:n].T\\n        for reg in (.0003,.001,.003):\\n            C=cho_factor(K[:n]+reg*np.eye(n)); a=cho_solve(C,t); E=a/np.diag(cho_solve(C,np.eye(n)))[:,None]; state.append((np.mean(abs(E)),K[n:]@a,t-E))\\n    state.sort(key=lambda a:a[0]); st=np.mean([a[1] for a in state[:7]],axis=0); os=np.mean([a[2] for a in state[:7]],axis=0)\\n    return out,loo,st,os\\n\\ndef _readout(study,x):\\n    from scipy.optimize import nnls\\n    out,loo,st,os=_portfolio(study,x); tr=study[\'train\']; y=np.asarray(tr[\'y\']); n=len(y)\\n    ks=sorted(out); oof=np.column_stack([loo[k][:,0] for k in ks]); pred=np.column_stack([out[k][:,0] for k in ks]); costs=np.mean(abs(oof-y[:,None]),axis=0)\\n    w=np.exp(-(costs-costs.min())/.012);w/=w.sum(); p=pred@w; op=oof@w\\n    # Predict nonlinear readout using a temporal amplitude as an additional proxy.\\n    t=tr[\'traj\']; dt=t.mean(0); _,_,V=np.linalg.svd(t-dt,full_matrices=False)\\n    U=(t-dt)@V[:3].T; Up=(os-dt)@V[:3].T; T=(st-dt)@V[:3].T; scale=U.std(0).clip(.035)\\n    # A strongly regularized mixture avoids relying on one family\'s response to tail inputs.\\n    P=np.c_[oof, op, Up[:,0], np.ones(n)]; Q=np.c_[pred,p,T[:,0],np.ones(len(x)-n)]\\n    anchor=np.r_[w,0.,0.,0.]; pen=np.diag(np.r_[np.full(len(w)+1,.24),.45,.24]); coef=np.linalg.solve(P.T@P+pen,P.T@y+pen@anchor)\\n    return .7*p+.3*(Q@coef),st\\n\\ndef solve_batch(study: dict) -> dict:\\n    tr=study[\'train\']; x=_calibrate(study[\'panel_a\'],np.r_[tr[\'x\'],study[\'test\'][\'x\']]); y,p=_readout(study,x)\\n    return {\'y_hat\':y,\'state_pred\':p,\'diagnostics_vec\':np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])}\\n\\n# Candidate readout operators fitted independently of the held-out responses.\\ndef _operator_library(study, x):\\n    n=len(x); one=np.ones(n); ii,jj=np.triu_indices(4); ia,ja=np.triu_indices(4,1)\\n    from itertools import combinations_with_replacement\\n    trip=np.array(list(combinations_with_replacement(range(4),3)))\\n    def poly(z, deg=2):\\n        a=np.c_[one,z,z[:,ii]*z[:,jj]]\\n        return np.c_[a,np.prod(z[:,trip],axis=2)] if deg==3 else a\\n    f,q,B,lin=_panel_features(study[\'panel_b\']); g=f(x,q)\\n    h=(np.c_[one,x,g] if lin else np.c_[one,g])@B\\n    pc=study[\'panel_c\']; u=np.asarray(pc[\'u\']); C=[]\\n    for k,forms in [(\'v1\',[lambda a,p:a/(1+p*abs(a)),lambda a,p:np.arctan(p*a),lambda a,p:np.arcsinh(p*a),lambda a,p:np.tanh(p*a)]),(\'v2\',[lambda a,p:np.logaddexp(p*a,-p*a)-np.log(2.),lambda a,p:a*a/(1+p*a*a)])]:\\n        v=pc[k]; trials=[]\\n        for fun in forms:\\n            for additive in (True,False):\\n                def mat(a,p):\\n                    w=fun(a,p[0]);return np.c_[np.ones(len(a)),a,w] if additive else np.c_[np.ones(len(a)),a+p[1]*w]\\n                def res(p):\\n                    M=mat(u,p);return (M@np.linalg.lstsq(M,v,rcond=None)[0]-v).ravel()\\n                for p0 in ([.25,.55],[1.1,.9]):\\n                    sol=least_squares(res,p0,bounds=([-0.45,-3],[4,3]),max_nfev=70)\\n                    M=mat(u,sol.x); pin=np.linalg.pinv(M); beta=pin@v; err=np.mean(((M@beta-v)/np.maximum(.04,1-np.sum(M*pin.T,axis=1))[:,None])**2)\\n                    trials.append((err,mat(x,sol.x)@beta))\\n        C.append(min(trials,key=lambda a:a[0])[1])\\n    yield \'panel\',poly(h);yield \'panel\',poly(C[0]);yield \'panel\',poly(C[1])\\n    maps=[np.tanh, np.arctan, lambda a:a/np.sqrt(1+a*a), erf,lambda a:a/(1+abs(a))]\\n    for fun in maps:\\n        for q in (1.0,1.25,1.5,1.75,2.0):\\n            z=fun(q*x); XZ=(x[:,:,None]*z[:,None,:]).reshape(n,-1)\\n            p=poly(z); P=np.c_[p,x,XZ]; delta=x-z/q\\n            yield \'direct\',P\\n            yield \'direct\',np.c_[p,x,XZ[:,[0,5,10,15]]]\\n            yield \'cubic\',np.c_[P,.4*np.prod(z[:,trip],axis=2)]\\n            yield \'cubic\',np.c_[P,.18*(z[:,ia]*z[:,ja])*(z[:,ia]-z[:,ja]),.35*z**3,.3*delta**2]\\n            for a in (h,C[0],C[1]):\\n                tilt=fun(a); mix=(z[:,:,None]*tilt[:,None,:]).reshape(n,-1)\\n                yield \'compose\',np.c_[P,.4*tilt,.35*tilt[:,ii]*tilt[:,jj],.25*mix]\\n            # A local quadratic in the two channels permits weak mixed cubic terms.\\n            yield \'coordinate\',np.c_[P,.28*(delta[:,:,None]*delta[:,None,:]).reshape(n,-1),.28*x*x*z,.22*x*z*z]\\n            for cut in (1.25,1.5):\\n                w=np.clip(x,-cut,cut); zc=fun(q*w)\\n                yield \'bounded\',np.c_[poly(zc),w,(w[:,:,None]*zc[:,None,:]).reshape(n,-1)]\\n    for mode in (1,2,3):\\n        z=np.sign(g)*abs(g)**(1/mode); XZ=(x[:,:,None]*z[:,None,:]).reshape(n,-1)\\n        yield \'operator\',np.c_[poly(z),x,XZ]\\n        yield \'operator\',np.c_[poly(z),x,XZ,.25*np.prod(z[:,trip],axis=2)]\\n\\n\\ndef _fit_readout_operators(study,x):\\n    y=np.asarray(study[\'train\'][\'y\']); n=len(y); eye=np.eye(n)\\n    rng=np.random.default_rng(20); perm=[rng.permutation(n) for _ in range(5)]\\n    splits=[np.array(a) for p in perm for a in np.array_split(p,5)]\\n    tail=np.argsort(np.linalg.norm(x[:n],axis=1))[-max(n//5,3):]\\n    best={}\\n    for kind,F in _operator_library(study,x):\\n        K=F@F[:n].T; G=K[:n]\\n        for reg in (.0004,.0018,.006,.022):\\n            C=cho_factor(G+reg*eye); inv=cho_solve(C,eye); a=inv@y\\n            E=np.zeros((len(perm),n))\\n            for i,j in enumerate(splits):\\n                E[i//5,j]=np.linalg.solve(inv[np.ix_(j,j)],a[j])\\n            score=.88*np.mean(abs(E))+.12*np.mean(abs(E[:,tail])); best.setdefault(kind,[]).append((score,K[n:]@a))\\n    R=[]\\n    for items in best.values():\\n        items.sort(key=lambda o:o[0]); keep=items[:5]; err=np.array([g[0] for g in keep]); w=np.exp(-(err-err[0])/.004);w/=w.sum();R.append((w@err,w@np.array([g[1] for g in keep])))\\n    R.sort(key=lambda o:o[0]); keep=R[:4]; err=np.array([g[0] for g in keep]); w=np.exp(-(err-err[0])/.006);w/=w.sum();return w@np.array([g[1] for g in keep])\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    tr=study[\'train\']; x=_calibrate(study[\'panel_a\'],np.r_[tr[\'x\'],study[\'test\'][\'x\']]); y,p=_readout(study,x)\\n    y=(1-0.5)*y+0.5*_fit_readout_operators(study,x)\\n    return {\'y_hat\':y,\'state_pred\':p,\'diagnostics_vec\':np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])}\\n"}'

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
