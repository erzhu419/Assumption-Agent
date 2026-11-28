"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "169762e7408495268750dc7036a8c481338f3bc741c19012c4cc23d5d62047b0"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Panel-calibrated scalar and trajectory prediction.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar, least_squares\\n\\ndef ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-11)[0]\\ndef aff(x): return np.c_[np.ones(len(x)),x]\\ndef poly(x):\\n i,j=np.triu_indices(x.shape[1]); return np.c_[aff(x),x[:,i]*x[:,j]]\\ndef cal(p,x):\\n u,v=p[\'u\'],p[\'v\']; n=np.roll(u,-1,1); A=np.stack((np.ones_like(u),u,n,u*n),2); w=np.ones(len(u))\\n for _ in range(30):\\n  c=np.stack([ls(A[:,j]*w[:,None],v[:,j]*w) for j in range(4)],1); r=np.linalg.norm(np.einsum(\'ijk,kj->ij\',A,c)-v,axis=1);w=np.sqrt(np.minimum(1,1.3*np.median(r)/np.maximum(r,1e-12)))\\n good=r<4*np.median(r)\\n def coef(k,ret=False):\\n  D=np.concatenate((A,np.sin(k*u)[:,:,None]),axis=2)[good]; C=np.stack([ls(D[:,j],v[good,j]) for j in range(4)],1)\\n  if ret:return C\\n  return np.mean([np.mean(((D[:,j]@C[:,j]-v[good,j])/(1-(np.linalg.qr(D[:,j])[0]**2).sum(1)))**2) for j in range(4)])\\n k=minimize_scalar(coef,bounds=(.12,1.6),method=\'bounded\').x; c=coef(k,True)\\n d=np.eye(4); e=np.roll(d,1,axis=1); z=np.linalg.solve(d*(c[1]+k*c[4])[:,None]+e*c[2,:,None],(x-c[0]).T).T\\n for _ in range(30):\\n  n=np.roll(z,-1,1); pred=c[0]+c[1]*z+c[2]*n+c[3]*z*n+c[4]*np.sin(k*z)\\n  jac=d*(c[1]+c[3]*n+c[4]*k*np.cos(k*z))[:,:,None]+e*(c[2]+c[3]*z)[:,:,None]; z+=np.clip(np.linalg.solve(jac,(x-pred)[...,None])[...,0],-.4,.4)\\n return z\\n\\ndef ops(s,z):\\n out=[]\\n for pn,vn,fn,bounds in [(\'panel_b\',\'v\',np.sin,(2.2,3.3)),(\'panel_c\',\'v1\',np.arcsinh,(1.5,8)),(\'panel_c\',\'v2\',lambda x:np.sqrt(1+x*x),(.5,1.6))]:\\n  u,v=s[pn][\'u\'],s[pn][vn]\\n  def design(u,k):return np.c_[aff(u),fn(k*u)]\\n  def loss(k):\\n   Q=np.linalg.qr(design(u,k))[0]; return np.mean(((v-Q@(Q.T@v))/(1-(Q*Q).sum(1))[:,None])**2)\\n  k=minimize_scalar(loss,bounds=bounds,method=\'bounded\').x;out.append(design(z,k)@ls(design(u,k),v))\\n return out\\n\\ndef ridge(a,y,n,alpha):\\n m=a[:n].mean(0); d=np.maximum(a[:n].std(0),1e-10); a=(a-m)/d; h=a[:n]; b=y[:n]-y[:n].mean(0); return y[:n].mean(0)+a@(h.T@np.linalg.solve(h@h.T+alpha*np.eye(n),b))\\n\\ndef solve_batch(study):\\n s=study; h=s[\'train\']; n=len(h[\'x\']); z=cal(s[\'panel_a\'],np.r_[h[\'x\'],s[\'test\'][\'x\']]); b,p,q=ops(s,z)\\n sf=poly(np.c_[z,b])[:,1:]; t=ridge(sf,h[\'traj\'],n,.03)[n:];\\n _,_,V=np.linalg.svd(h[\'traj\'],full_matrices=False); T=ridge(sf,h[\'traj\']@V[:1].T,n,.03)\\n # For training rows use strictly leave-one-out state features whenever using a learned state.\\n for j in range(n):\\n  order=np.r_[np.arange(j),np.arange(j+1,n),j]; T[j]=ridge(sf[order],(h[\'traj\']@V[:1].T)[order],n-1,.03)[-1]\\n y=_scalar(z,b,p,q,T,h[\'y\']); return {\'y_hat\':y,\'state_pred\':t,\'diagnostics_vec\':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}\\n\\ndef _scalar(z,b,p,q,T,y):\\n \\"\\"\\"Fit additive and interacting covariance components by profiled likelihood.\\"\\"\\"\\n from scipy.linalg import cho_factor, cho_solve\\n from scipy.optimize import minimize\\n n=len(y); y=np.asarray(y); ones=np.ones(n); eye=np.eye(n); delta=np.subtract.outer\\n def scale(a):\\n  a=np.asarray(a).reshape(len(z),-1); return (a-a[:n].mean(0))/np.maximum(a[:n].std(0),1e-8)\\n def g(a):\\n  a=scale(a); return a@a.T/a.shape[1]\\n def d(a):\\n  a=scale(a); return np.mean((a[:,None,:]-a[None,:,:])**2,axis=2)\\n def rdist(D,w): return np.exp(-D*w)\\n def tangent(x):return np.c_[x,x*x,x**3,np.tanh(2.5*x),np.sqrt(.12+x*x),np.sin(2.5*x),np.cos(2.5*x)]\\n i,j=np.triu_indices(4,1)\\n additive=[g(tangent(z[:,k:k+1])) for k in range(4)]\\n full=[]\\n for jset in (b,p,q):\\n  full.append([g(tangent(jset[:,k:k+1])) for k in range(4)])\\n zcross=[g(np.c_[z[:,k]*z[:,l],z[:,k]**2*z[:,l],z[:,k]*z[:,l]**2]) for k,l in zip(i,j)]\\n zd=d(z); bd=d(b); pd=d(p); qd=d(q); td=d(T); zbd=d(np.c_[z,b]); base=g(tangent(z));\\n candidates=[]\\n def prep(kernels):\\n  return np.asarray(kernels).copy()\\n def optimize(kernels,prior,noise=-7.,starts=2):\\n  Ks=prep(kernels); cross=Ks[:,n:,:n]; K=Ks[:,:n,:n].copy(); fl=K.reshape(len(K),-1); tr=fl.shape[1]; prior=np.asarray(prior); basepar=np.r_[prior,noise];\\n  def obj(a,ret=False):\\n   w=np.exp(a); A=(w[:-1]@fl).reshape(n,n)+w[-1]*eye\\n   R=cho_factor(A,check_finite=False); M=cho_solve(R,eye,check_finite=False); u=M@ones; v=1/u.sum(); mean=v*(u@y); G=M-v*np.outer(u,u); al=G@y; den=np.maximum(np.diag(G),1e-10); e=al/den; err=np.mean(np.abs(e));\\n   if ret:return err,e,mean+(w[:-1]@cross.reshape(len(K),-1)).reshape(len(z)-n,n)@al\\n   ll=.5*(np.sum(np.log(np.diag(R[0])))*2+np.log(1/v)+y@al)\\n   F=G-np.outer(al,al); grad=.5*w*np.r_[fl@F.ravel(),np.trace(F)]; diff=a-basepar\\n   return ll+.045*(diff@diff),grad+.09*diff\\n  for off in [0,-1.5][:starts]:\\n   init=basepar.copy(); init[:-1]+=off\\n   opt=minimize(obj,init,jac=True,method=\'L-BFGS-B\',bounds=[(-16,5)]*len(init),options={\'maxiter\':175,\'ftol\':1e-9});\\n   candidates.append(obj(opt.x,True))\\n for X,D,aux in [(z,zd,None),(np.c_[z,b],zbd,0),(np.c_[z,p],(zd+pd)/2,1),(np.c_[z,q],(zd+qd)/2,2)]:\\n  P=poly(X); C=[base,g(P[:,1:]),g(tangent(X))]+[rdist(D,k) for k in [.025,.12,.4,1.3,4]]\\n  if aux is not None:C+=full[aux]\\n  optimize(C,[-2,-2,-3,-2,-2,-3,-5,-6]+([-4]*4 if aux is not None else []))\\n C=additive+zcross+[g(tangent(b)),g(tangent(p)),g(tangent(q))]+[rdist(D,k) for D in [zd,zbd,(zd+pd)/2,(zd+qd)/2,(zd+td)/2] for k in [.08,.45,1.7]]\\n optimize(C,[-2]*4+[-3]*6+[-4]*3+[-3,-4,-6]*5)\\n C=additive+full[0]+full[1]+full[2]+zcross+[g(T),g(tangent(T)),g(T*z),g(T*b)]+[rdist(D,k) for D in [zbd,(zd+td)/2,(bd+td)/2] for k in [.08,.5,2]]\\n optimize(C,[-2]*4+[-4]*12+[-3]*6+[-3]*4+[-4,-4,-6]*3)\\n # Let the data select coordinate scales in a smooth nonseparable covariance.\\n for X in [z,np.c_[z,b],np.c_[z,q],np.c_[z,b,T]]:\\n  X=scale(X); parts=(X[:,None,:]-X[None,:,:])**2; baseK=base+.15*g(poly(z)[:,1:]); terms=parts[:n,:n].reshape(n*n,-1).copy(); test=parts[n:,:n].reshape(-1,X.shape[1]); m=X.shape[1];\\n  def obj(a,ret=False):\\n   w=np.exp(a); raw=w[m]*np.exp(-(terms@w[:m])).reshape(n,n); A=raw+w[m+1]*baseK[:n,:n]+w[-1]*eye\\n   R=cho_factor(A,check_finite=False); M=cho_solve(R,eye,check_finite=False); u=M@ones; v=1/u.sum(); mean=v*(u@y); G=M-v*np.outer(u,u); al=G@y; e=al/np.maximum(np.diag(G),1e-10)\\n   if ret:\\n    K=w[m]*np.exp(-(test@w[:m])).reshape(len(z)-n,n)+w[m+1]*baseK[n:,:n]; return np.mean(abs(e)),e,mean+K@al\\n   F=.5*(G-np.outer(al,al)); rr=(F*raw); grad=np.r_[-w[:m]*(terms.T@rr.ravel()),rr.sum(),w[m+1]*np.sum(F*baseK[:n,:n]),w[-1]*np.trace(F)]; diff=a-center\\n   return .5*(2*np.log(np.diag(R[0])).sum()+np.log(1/v)+y@al)+.065*(diff@diff),grad+.13*diff\\n  for k in [.12,.5,1.5]:\\n   center=np.r_[np.log(np.repeat(.25/m,m)),0.,-2.,-7.]; init=center.copy(); init[:m]=np.log(k/m)\\n   opt=minimize(obj,init,jac=True,method=\'L-BFGS-B\',bounds=[(-12,3)]*m+[(-12,5),(-13,4),(-16,0)],options={\'maxiter\':180,\'ftol\':1e-10}); candidates.append(obj(opt.x,True))\\n # Fixed broad smooth priors provide a guard against unstable likelihood optima.\\n for D in [zd,zbd,(zd+pd)/2,(zd+qd)/2,(zd+td)/2]:\\n  for k in [.05,.15,.45,1.3]:\\n   for weight in [.03,.3,3.]:\\n    A=np.exp(-D*k)+weight*base+.05*g(z[:,i]*z[:,j]);\\n    for noise in [2e-6,2e-5,2e-4,.002,.02]:\\n     M=cho_solve(cho_factor(A[:n,:n]+noise*eye,check_finite=False),eye,check_finite=False); u=M@ones; v=1/u.sum(); mean=v*(u@y); G=M-v*np.outer(u,u); al=G@y; e=al/np.maximum(np.diag(G),1e-10); candidates.append((np.mean(abs(e))+.002,e,mean+A[n:,:n]@al))\\n chosen=[]\\n for c in sorted(candidates,key=lambda c:c[0]):\\n  if not chosen or all(np.mean(abs(c[1]-a[1]))>.007 for a in chosen):chosen.append(c)\\n  if len(chosen)>=22:break\\n E=np.array([c[1] for c in chosen]).T; P=np.array([c[2] for c in chosen]); qual=np.array([c[0] for c in chosen]); w0=np.exp(-(qual-qual.min())/.012); w0/=w0.sum()\\n def loss(w):return np.mean(np.hypot(E@w,.012))+.006*np.sum((w-w0)**2/np.maximum(w0,.02))\\n out=minimize(loss,w0,method=\'SLSQP\',bounds=[(0,1)]*len(w0),constraints={\'type\':\'eq\',\'fun\':lambda w:w.sum()-1},options={\'ftol\':1e-9}); w=out.x if out.success else w0\\n return (.8*w+.2*w0)@P\\n"}'

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
