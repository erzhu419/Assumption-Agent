"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "c5ab6cf040d128802ab3c991517576da0019419b8025ba2d7b9837bccd710672"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nimport pickle\\nfrom scipy.linalg import cho_factor,cho_solve\\nfrom itertools import product\\nfrom scipy.optimize import minimize\\nfrom scipy.spatial.distance import cdist\\n\\ndef calibrate(p):\\n u,v=p[\'u\'],p[\'v\']; i,j=np.triu_indices(len(u),1); d=u[i]-u[j]; scale=np.median((v[i]-v[j])/np.where(abs(d)>1e-10,d,np.inf),axis=0); offset=np.median(v-u*scale,axis=0)\\n for _ in range(3):\\n  r=np.linalg.norm(v-u*scale-offset,axis=1); keep=r<max(6*np.median(r),.0005)\\n  if keep.sum()<6: break\\n  for j in range(u.shape[1]): scale[j],offset[j]=np.linalg.lstsq(np.c_[u[keep,j],np.ones(keep.sum())],v[keep,j],rcond=None)[0]\\n return scale,offset\\n\\ndef coords(s,x):\\n a,b=calibrate(s[\'panel_a\']); z=(x-b)/a; p=s[\'panel_b\']; t=np.linalg.lstsq(np.c_[np.ones(len(p[\'u\'])),p[\'u\']],p[\'v\'],rcond=None)[0]; return [x,z,z@t[1:]+t[0],x@t[1:]+t[0]]\\n\\ndef fitgp(z,q,y,kind,objective=\'ml\',sig=.009):\\n n,d=z.shape; N=n+len(q); Z=np.vstack([z,q]); D=Z[:,None,:]-Z[None,:,:]; D2=D**2; I=np.eye(n); mu=y.mean(axis=0); y=(y-mu).reshape(n,-1); dot=Z@Z.T; B0=np.ones((N,N)); B1=dot; B2=dot**2\\n full=kind in (\'full\',\'fulladd\',\'fullm52\'); add=kind in (\'add\',\'fulladd\',\'diagadd\'); isot=kind in (\'iso\',\'add\'); m52=kind==\'fullm52\'; ij=np.tril_indices(d,-1); nf=len(ij[0]) if full else 0; nd=1 if isot else d; nk=5 if add else 4\\n def cov(p,allrows=False):\\n  R=D if allrows else D[:n,:n]; T=D2 if allrows else D2[:n,:n]; sl=np.s_[:,:] if allrows else np.s_[:n,:n]; v=np.exp(p[:nd]); W=R*v; G=W\\n  if full:\\n   L=np.diag(v); L[ij]=p[nd:nd+nf]; G=R@L\\n  S=np.sum(G**2,axis=-1)\\n  if m52:\\n   A=np.sqrt(5*S); K=(1+A+5*S/3)*np.exp(-A); gradk=-(5/6)*(1+A)*np.exp(-A)\\n  else: K=np.exp(-S); gradk=-K\\n  bases=[B0[sl],B1[sl],B2[sl],K]; parts=None\\n  if add:\\n   parts=np.exp(-T*v**2); bases.append(parts.mean(axis=-1))\\n  amps=np.exp(p[nd+nf:nd+nf+nk]); C=sum(a*b for a,b in zip(amps,bases));\\n  if allrows:return C\\n  noise=np.exp(p[-1]); C=C+noise*I; J=2*amps[3]*gradk[:,:,None]*G*W\\n  if add:J-=2*amps[4]/d*parts*T*v**2\\n  deriv=[J.sum(axis=-1)] if isot else list(J.transpose(2,0,1));\\n  if full: deriv += [2*amps[3]*gradk*G[:,:,j]*R[:,:,i] for i,j in zip(*ij)]\\n  return C,np.stack(deriv+[a*b for a,b in zip(amps,bases)]+[noise*I])\\n def loss(p):\\n  C,g=cov(p); CF=cho_factor(C); alpha=cho_solve(CF,y); inv=cho_solve(CF,I)\\n  if objective==\'ml\':\\n   f=.5*np.sum(alpha*y)+y.shape[1]*np.log(CF[0].diagonal()).sum(); H=.5*(y.shape[1]*inv-alpha@alpha.T)\\n  else:\\n   a=inv.diagonal()[:,None]; e=alpha/a; J=(inv@((e/a)@alpha.T)); V=(e*e/a).sum(axis=1); f=.5*np.sum(e*e)*n; H=((inv*V)@inv-.5*(J+J.T))*n\\n  diff=p-p0; return f+.09*np.sum(diff*diff), np.einsum(\'ijk,jk->i\',g,H)+.18*diff\\n p0=np.r_[np.full(nd,-.55),np.zeros(nf),np.log([1,.5,.025,1]+([.25] if add else [])),2*np.log(sig)]; bounds=[(-4,1)]*nd+[(-2,2)]*nf+[(-11,4)]*nk+[(np.log(sig**2)-2,np.log(sig**2)+3)]; opt=minimize(loss,p0,jac=True,bounds=bounds,method=\'L-BFGS-B\',options={\'maxiter\':115,\'ftol\':1e-8}); p=opt.x\\n C=cov(p,True); pred=C[n:,:n]@cho_solve(cho_factor(C[:n,:n]+np.exp(p[-1])*I),y)+mu\\n return pred,p,opt.fun\\n\\n\\ndef _baseline_scalar(z, q, y):\\n    \\"\\"\\"Average smooth, soft-sign, and pairwise spatial response models.\\"\\"\\"\\n    Z=np.concatenate((z,q)); n=len(z)\\n    D=(Z[:,None,:]-Z[None,:,:])**2\\n    L=Z@Z.T\\n    F=Z/(1+.83*np.abs(Z)); G=F@F.T\\n    A=np.exp(-.44*D)\\n    pairs=sum(A[:,:,i]*A[:,:,j] for i in range(Z.shape[1]) for j in range(i))\\n    smooth=1+.35*L+np.exp(-.44*D.sum(axis=-1))\\n    def predict(C,r):\\n        return C[n:,:n]@cho_solve(cho_factor(C[:n,:n]+r*np.eye(n)),y)\\n    return (.66*predict(smooth+.2*(1+G)*(1+L),.0001)\\n            +.26*predict(smooth+.2*pairs*(1+L),.0001)\\n            +.08*predict(1+.4*L+np.exp(-.25*D.sum(axis=-1)),.000003))\\n\\ndef _chart(z, mode, h):\\n    if mode==1:return np.arcsinh(h*z)/h\\n    if mode==2:return z/(1+h*abs(z))\\n    if mode==3:return np.sign(z)*np.log1p(h*abs(z))/h\\n    if mode==4:return z/(1+h*z*z)\\n    return z\\n\\ndef _gram(z, spec):\\n    mode,h,kind,a,degree,b,c=spec\\n    q=_chart(z,mode,h)\\n    L=z@z.T; M=q@q.T; D=cdist(q,q,\'sqeuclidean\'); D0=cdist(z,z,\'sqeuclidean\')\\n    U=(1+.38*L)**degree\\n    if kind==0: K=np.exp(-a*D)\\n    elif kind==1: K=(1+a*D)**(-1)\\n    elif kind==2:\\n        R=np.sqrt(5*a*D); K=(1+R+R**2/3)*np.exp(-R)\\n    elif kind==3: K=np.exp(-a*D)*(1+b*L); b=0\\n    elif kind==4:\\n        A=np.exp(-a*(q[:,None,:]-q[None,:,:])**2); S=A.sum(2); K=(S*S-(A*A).sum(2))/12*(1+b*L); b=0\\n    elif kind==5: K=np.exp(-a*D0)+b*(1+M)*(1+L); b=0\\n    elif kind==6: K=np.exp(-a*D0)+b*((1+M)**2+.2*M*L); b=0\\n    elif kind==7: K=np.exp(-a*D0)+b*np.exp(-a*D); b=0\\n    elif kind==8:\\n        V=1/(1+c*np.sum(q*q,axis=1)); K=np.exp(-a*D0)+b*(1+.55*M)**3*V[:,None]*V[None,:]; b=0\\n    elif kind==9:\\n        V=np.array([v for v in product([-1,0,1],repeat=z.shape[1]) if any(v) and next(t for t in v if t)!=-1]); V=V/np.linalg.norm(V,axis=1)[:,None]; P=q@V.T\\n        S=np.sin(a*P); C=np.cos(a*P); K=np.exp(-c*D0)+b*(S@S.T+C@C.T)*(1+.4*L)/len(V); b=0\\n    elif kind==10:\\n        G=np.sqrt(np.maximum(D0,0)); K=np.exp(-a*D0)+b*np.exp(-c*G)*(1+.25*L); b=0\\n    else:\\n        P=np.tanh(h*z); J=P@P.T; K=np.exp(-a*D0)+b*(1+J)*(1+L); b=0\\n    return U+K+b*M\\n\\ndef _prediction(C,i,j,y,r):\\n    return C[np.ix_(j,i)]@cho_solve(cho_factor(C[np.ix_(i,i)]+r*np.eye(len(i))),y)\\n\\ndef solve_batch(study: dict) -> dict:\\n    train=study[\'train\']; x=np.asarray(train[\'x\']); q=np.asarray(study[\'test\'][\'x\']); T=np.asarray(train[\'traj\']); n=len(x)\\n    if not len(q): return {\'y_hat\':np.empty(0),\'state_pred\':np.empty((0,len(study[\'eval_times\']))),\'diagnostics_vec\':np.zeros(4)}\\n    Z=coords(study,np.concatenate((x,q)))[2]\\n    charts=coords(study,np.concatenate((x,q)))\\n    y=np.asarray(train[\'y\'])\\n    yp=np.zeros(len(q))\\n    for model, weight in [((2, (2, 0.22, 0, 0.46, 0, 0.18, 0.22), 4e-05), 0.4219488), ((2, (2, 1.4, 11, 0.23, 2, 0.8, 0.22), 4e-05), 0.2272032), ((1, (2, 0.22, 9, 2.25, 1, 0.8, 0.11), 4e-05), 0.123648), ((1, (0, 1, 9, 2.25, 0, 0.035, 0.11), 2e-06), 0.1472), ((3, (2, 1.4, 11, 0.23, 2, 0.8, 0.22), 0.00023), 0.08)]:\\n        if model is None:\\n            pred=_baseline_scalar(Z[:n],Z[n:],y)\\n        else:\\n            index,spec,r=model\\n            pred=_prediction(_gram(charts[index],spec),np.arange(n),np.arange(n,n+len(q)),y,r)\\n        yp+=weight*pred\\n    _,_,V=np.linalg.svd(T,full_matrices=False); V=V[:2]; A=T@V.T\\n    scales=np.maximum(A.std(axis=0),1e-8)\\n    P=fitgp(Z[:n],Z[n:],A/scales,\'full\',\'ml\',.016)[0]\\n    state=(P*scales)@V\\n    return {\'y_hat\':yp,\'state_pred\':state,\'diagnostics_vec\':np.array([yp.mean(),yp.std(),state[:,-1].mean(),state[:,-1].std()])}\\n"}'

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
