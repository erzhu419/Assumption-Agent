"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "10cedbe8fbb1be5f328579cb2e700a3a5a45caa48ebe09e689b29ae0b7d9a35f", "node_01": "ed3b04643549e29a41ef9d2e5a98262b90c0d87bc3a926938332719b9b35be6d"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover the shared latent coordinates from the reference calibration.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    panel = study[\'panel_a\']\\n    u, v = np.asarray(panel[\'u\'], float), np.asarray(panel[\'v\'], float)\\n    d = u.shape[1]\\n    nxt = (np.arange(d) + 1) % d\\n\\n    def terms(q, k):\\n        return np.column_stack((np.ones(len(q)), q, q[:, k]**3, q[:, k]*q[:, nxt[k]]))\\n\\n    # Gross reference corruptions affect whole rows. Refine the joint inlier\\n    # mask after fitting the sparse cyclic model to the uncorrupted majority.\\n    delta = np.linalg.norm(v-u, axis=1)\\n    keep = delta <= max(3., 4.*np.median(delta))\\n    coef = np.stack([np.linalg.lstsq(terms(u[keep], k), v[keep,k], rcond=None)[0] for k in range(d)], axis=1)\\n\\n    def forward(q):\\n        return coef[0] + q @ coef[1:d+1] + coef[-2]*q**3 + coef[-1]*q*q[nxt]\\n\\n    errors = np.linalg.norm(np.array([forward(q) for q in u])-v, axis=1)\\n    good = errors <= max(0.003, 6.*np.median(errors))\\n    if np.count_nonzero(good) >= d+3:\\n        coef = np.stack([np.linalg.lstsq(terms(u[good], k), v[good,k], rcond=None)[0] for k in range(d)], axis=1)\\n\\n    def jac(q):\\n        out = coef[1:d+1].T.copy()\\n        k = np.arange(d)\\n        out[k,k] += 3.*coef[-2]*q*q + coef[-1]*q[nxt]\\n        out[k,nxt] += coef[-1]*q\\n        return out\\n\\n    train = np.asarray(study[\'train\'][\'x\'], float)\\n    test = np.asarray(study[\'test\'][\'x\'], float)\\n    x = np.concatenate((train, test), axis=0)\\n    init = np.linalg.lstsq(coef[1:d+1].T, (x-coef[0]).T, rcond=None)[0].T\\n    latent = np.empty_like(x)\\n    for i, (target, q) in enumerate(zip(x, init)):\\n        latent[i] = least_squares(lambda z: forward(z)-target, q, jac=jac, max_nfev=50,\\n                                  ftol=1e-11, xtol=1e-11, gtol=1e-11).x\\n    return {\'cubic_train\': latent[:len(train)]**3, \'cubic_test\': latent[len(train):]**3}\\n", "node_01": "\\"\\"\\"Profile calibration models from the reference panel, then fit joint readouts.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar, least_squares\\nfrom scipy.linalg import solve\\n\\n\\ndef _poly(x, degree):\\n    return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for d in range(1,degree+1) for i in combinations_with_replacement(range(x.shape[1]),d)])\\n\\n\\ndef _cal_design(x, kind, k, j):\\n    a=_poly(x,1)\\n    if kind==\'cyclic\': return np.c_[a,x[:,j]**3,x[:,j]*x[:,(j+1)%x.shape[1]]]\\n    if kind==\'cube\': return np.c_[a,x**3]\\n    if kind==\'quadratic\': return _poly(x,2)\\n    if kind==\'cubic\': return _poly(x,3)\\n    if kind==\'sin\': return np.c_[a,np.sin(k*x)]\\n    if kind==\'cos\': return np.c_[a,np.cos(k*x)]\\n    if kind==\'both\': return np.c_[a,np.sin(k*x),np.cos(k*x)]\\n    if kind==\'diag\': return np.c_[a,np.sin(k*x[:,j]),np.cos(k*x[:,j]),x[:,j]*x[:,(j+1)%x.shape[1]]]\\n    if kind==\'atan\': return np.c_[a,np.arctan(k*x)]\\n    return np.c_[a,np.tanh(k*x)]\\n\\n\\ndef _calibrate(u,v,x,z):\\n    d=u.shape[1]; n=len(u); ii=np.arange(n)\\n    def fit(kind,k,t):\\n        mats=[_cal_design(u,kind,k,j) for j in range(d)]\\n        keep=t[np.argsort(np.linalg.norm(v[t]-u[t],axis=1))[:max(1,int(.82*len(t)))]]\\n        for step in range(5):\\n            w=[np.linalg.lstsq(m[keep],v[keep,j],rcond=1e-9)[0] for j,m in enumerate(mats)]\\n            r=np.linalg.norm(np.column_stack([m@c for m,c in zip(mats,w)])-v,axis=1)\\n            nxt=t[np.argsort(r[t])[:len(keep)]]\\n            if np.array_equal(np.sort(nxt),np.sort(keep)): break\\n            keep=nxt\\n        return np.median(r[t]),w,r\\n    folds=np.array_split(np.random.default_rng(251).permutation(n),4)\\n    candidates=[]\\n    for kind in (\'cyclic\',\'cube\',\'quadratic\',\'cubic\',\'sin\',\'cos\',\'both\',\'diag\',\'atan\',\'tanh\'):\\n        if kind in (\'cyclic\',\'cube\',\'quadratic\',\'cubic\'): k=1.\\n        else:\\n            gs=np.linspace(.5,5.,37); m=np.argmin([fit(kind,g,ii)[0] for g in gs]); k=minimize_scalar(lambda g:fit(kind,g,ii)[0],bounds=(gs[max(0,m-1)],gs[min(m+1,len(gs)-1)]),method=\'bounded\').x\\n        err=np.mean([np.median(fit(kind,k,np.setdiff1d(ii,h))[2][h]) for h in folds]); e,w,_=fit(kind,k,ii)\\n        candidates.append((err,e,kind,k,w))\\n    out=[(\'frozen\',z)]\\n    for err,e,kind,k,w in sorted(candidates,key=lambda a:a[0])[:2]:\\n        def calc(p): return np.array([_cal_design(p[None,:],kind,k,j)[0]@w[j] for j in range(d)])\\n        zz=[]\\n        for a,b in zip(z,x):\\n            fit1=least_squares(lambda p:calc(p)-b,a,max_nfev=35)\\n            zz.append(fit1.x)\\n        out.append((kind,np.array(zz)))\\n    return out\\n\\n\\ndef _frequency(u,v,fun,lo=.25,hi=8.):\\n    def design(k):return np.c_[np.ones(len(u)),fun(k*u)]\\n    def err(k):\\n        a=design(k); return np.mean((a@np.linalg.lstsq(a,v,rcond=None)[0]-v)**2)\\n    g=np.linspace(lo,hi,125); j=np.argmin([err(k) for k in g]); f=minimize_scalar(err,bounds=(g[max(0,j-1)],g[min(j+1,len(g)-1)]),method=\'bounded\').x\\n    return f\\n\\n\\ndef _fit(a, t, j, y, p):\\n    m=a[t].mean(0); sd=np.maximum(a[t].std(0),1e-8); at=(a[t]-m)/sd; bt=(a[j]-m)/sd; mu=y[t].mean(0)\\n    return mu+bt@solve(at.T@at+p*len(t)*np.eye(at.shape[1]),at.T@(y[t]-mu),assume_a=\'pos\')\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s,z0=inputs[\'__input__\'],inputs[\'node_00\']; tr=s[\'train\']; x=np.vstack((tr[\'x\'],s[\'test\'][\'x\'])); z=np.cbrt(np.vstack((z0[\'cubic_train\'],z0[\'cubic_test\'])))\\n    b,c,a=s[\'panel_b\'],s[\'panel_c\'],s[\'panel_a\']; f=_frequency(b[\'u\'],b[\'v\'],np.cos); g=_frequency(c[\'u\'],c[\'v1\'],np.arctan)\\n    y=np.c_[tr[\'y\'],tr[\'traj\']]; n=len(y); ii=np.arange(n); test=np.arange(n,len(x)); folds=np.array_split(np.random.default_rng(232).permutation(n),5); cv=[(np.setdiff1d(ii,h),h) for h in folds]; results=[]\\n    for name,u in _calibrate(a[\'u\'],a[\'v\'],x,z)+[(\'raw\',x)]:\\n        cs,ss=np.cos(f*u),np.sin(f*u); at=np.arctan(g*u); cx=np.cos(f*x); sq=_poly(ss,2)[:,1+u.shape[1]:]; cq=_poly(cs,2)[:,1+u.shape[1]:]; j,k=np.triu_indices(u.shape[1],1)\\n        bases=[cs,np.c_[cs,ss],np.c_[cs,at],np.c_[cs,cx,at],np.c_[cs,at,sq],np.c_[cs,cx,at,sq],np.c_[cs,at,cq],np.c_[cs,at,u,u**3],np.c_[cs,at,np.cos(f*(u[:,j]+u[:,k])),np.cos(f*(u[:,j]-u[:,k]))],np.c_[cs,at,np.cos(f*u**3)],np.c_[cs,at,np.cos(g*u)],np.c_[cs,at,np.sin(g*u)]]\\n        for bidx,ab in enumerate(bases):\\n            for penalty in (.0001,.003,.025,.12):\\n                e=np.array([abs(_fit(ab,t,h,y,penalty)-y[h]).mean(0) for t,h in cv]); v=np.c_[e[:,0],e[:,1:].mean(1)]; metric=.8*v.mean(0)+.2*v.max(0)\\n                results.append((metric,(name,bidx),ab,penalty))\\n    out=[]\\n    for k in range(2):\\n        top={}\\n        for e,m,ab,p in sorted(results,key=lambda r:r[0][k]):\\n            if m not in top: top[m]=(e[k],_fit(ab,ii,test,y,p))\\n            if len(top)==4:break\\n        err,pred=zip(*top.values()); err=np.array(err); w=np.exp(-22*(err-err.min())/max(.01,err.min())); out.append(np.einsum(\'i,ijk->jk\',w/w.sum(),np.array(pred)))\\n    yh=out[0][:,0]; st=out[1][:,1:]\\n    return {\'y_hat\':yh,\'state_pred\':st,\'diagnostics_vec\':np.array([yh.mean(),yh.std(),st[:,-1].mean(),st[:,-1].std()])}\\n"}'

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
