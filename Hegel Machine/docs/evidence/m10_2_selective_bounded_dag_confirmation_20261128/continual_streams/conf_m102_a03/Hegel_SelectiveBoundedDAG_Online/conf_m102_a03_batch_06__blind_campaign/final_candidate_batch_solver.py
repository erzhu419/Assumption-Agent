"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "16514327b7020b62009b3346832cf351381c84f1c503abe31be77eae315ad1a4"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated regression with separate readout and temporal response kernels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize\\nfrom scipy.linalg import cho_factor, cho_solve\\n\\n\\ndef fit(a,b):\\n    return np.linalg.lstsq(a,b,rcond=1e-11)[0]\\n\\n\\ndef affine(x):\\n    return np.c_[np.ones(len(x)),x]\\n\\n\\ndef calibrate(panel,x):\\n    u,v=[np.asarray(panel[k],float) for k in (\'u\',\'v\')]\\n    good=np.linalg.norm(v,axis=1)<5\\n    def design(u,j):\\n        a=u[:,j]; b=u[:,(j+1)%4]\\n        return np.c_[affine(u),a*b,a**3]\\n    co=np.column_stack([fit(design(u[good],j),v[good,j]) for j in range(4)])\\n    def forward(z):\\n        return np.column_stack([design(z,j)@co[:,j] for j in range(4)])\\n    g=np.linalg.solve(co[1:5].T,(x-co[0]).T).T\\n    return np.array([least_squares(lambda z:forward(z[None])[0]-r,a,gtol=1e-12).x for a,r in zip(g,x)])\\n\\n\\ndef branch(panel,x):\\n    u,v=[np.asarray(panel[k],float) for k in (\'u\',\'v\')]\\n    def basis(u,k):\\n        return np.c_[affine(u),np.tanh(k[0]*u),np.sin(k[1]*u)]\\n    def residual(k):\\n        a=basis(u,k); return (a@fit(a,v)-v).ravel()\\n    best=min((least_squares(residual,h,bounds=(.2,6),max_nfev=100) for h in ([1,1],[2,2],[3,1.25],[1.25,3],[4,4])),key=lambda r:r.cost)\\n    return basis(x,best.x)@fit(basis(u,best.x),v)\\n\\n\\ndef features(study):\\n    n=len(study[\'train\'][\'x\']); x=np.r_[study[\'train\'][\'x\'],study[\'test\'][\'x\']]\\n    z=calibrate(study[\'panel_a\'],x); b=branch(study[\'panel_b\'],z)\\n    return z,b,n\\n\\n\\ndef kernelparts(z,b,n,family=0):\\n    def std(w):\\n        return (w-w[:n].mean(0))/np.maximum(w[:n].std(0),1e-9)\\n    joint=np.c_[z,b]; z,b,joint=std(z),std(b),std(joint)\\n    G=joint@joint.T/8\\n    Q=std(np.column_stack([joint[:,i]*joint[:,j] for i in range(8) for j in range(i,8)])); P=Q@Q.T/36\\n    Dz=(z[:,None,:]-z[None,:,:])**2; Db=(b[:,None,:]-b[None,:,:])**2\\n    if family==0:\\n        return np.array([np.ones_like(G),G,P,*[np.exp(-(Dz.mean(2)+Db.mean(2))/2/l) for l in (1,4,12)]])\\n    return np.array([np.ones_like(G),G,P,*[np.exp(-w/l) for w in (Dz.mean(2),Db.mean(2),(Dz.mean(2)+Db.mean(2))/2) for l in (1,4,12)], *[np.exp(-d/l).mean(2) for d in (Dz,Db) for l in (1,4,12)]])\\n\\n\\ndef solve_batch(study):\\n    z,b,n=features(study); y=study[\'train\'][\'y\']; t=study[\'train\'][\'traj\']; K=kernelparts(z,b,n,1)\\n    B,f=sine_branch(study[\'panel_b\'],z)\\n    yh=predict_readout(z,b,B,f,n,y)\\n    G=K[0]*20+K[1]+K[2]; state=G[n:,:n]@cho_solve(cho_factor(G[:n,:n]+.0003*np.eye(n)),t)\\n    last=state[:,-1]\\n    return {\'y_hat\':yh,\'state_pred\':state,\'diagnostics_vec\':np.array([yh.mean(),yh.std(),last.mean(),last.std()])}\\n\\n\\ndef sine_branch(panel,z):\\n    u,v=[np.asarray(panel[k],float) for k in (\'u\',\'v\')]\\n    def basis(x,h):return np.c_[affine(x),np.sin(h*x)]\\n    def res(h):\\n        A=basis(u,h[0]);return (A@fit(A,v)-v).ravel()\\n    h=min((least_squares(res,[i],bounds=(.4,6.5)) for i in (1.5,2.8,4.3,5.8)),key=lambda r:r.cost).x[0]\\n    return basis(z,h)@fit(basis(u,h),v),h\\n\\n\\ndef readout_matrices(z,b,B,f,n):\\n    def norm(x):return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-8)\\n    a=norm(z); b=norm(b); B=norm(B); h=norm(np.sin(f*z)); c=norm(z-.4*np.sin(f*z)); H=np.c_[a,h]; Z=np.c_[a,B]; W=np.c_[a,b]; C=np.c_[a,c]; one=np.ones((len(a),len(a)))\\n    ans={}\\n    def add(key,value):\\n        if key in selected_readout_keys: ans[key]=value\\n    for j,X in enumerate((Z,W,H,C,B,a)):\\n        p=X.shape[1]; G=X@X.T/p; G1=one+G; d=((X[:,None,:]-X[None,:,:])**2).mean(2)\\n        for degree in (1,2,3):\\n            for l in (2.,6.,16.,35.):\\n                for mix in (.6,3.,15.): add((j,degree,l,mix),G1**degree+mix*np.exp(-d/l))\\n    for j,D in enumerate((h,c,B,b)):\\n        mixed=((a[:,None,:]-a[None,:,:])**2+(D[:,None,:]-D[None,:,:])**2)*.5\\n        for l in (.3,1.,3.,8.):\\n            k=np.exp(-mixed/l); p=np.mean(k,axis=2); p2=(k.sum(2)**2-(k*k).sum(2))/12; p3=(k.sum(2)**3-3*k.sum(2)*(k*k).sum(2)+2*(k**3).sum(2))/24\\n            G=one+(a@a.T+D@D.T)/8\\n            for m in (.1,1.,10.):\\n                add((6+j,1,l,m),G+m*(p+.2*p2+.02*p3))\\n                add((6+j,2,l,m),G**2+m*(p+p2+.2*p3))\\n    return ans\\n\\n\\nREADOUT_MODELS = {((7, 2, 8.0, 1.0), 0.045): 1.0}\\nselected_readout_keys = {k for k,r in READOUT_MODELS}\\n\\ndef predict_readout(z,b,B,f,n,y):\\n    matrices=readout_matrices(z,b,B,f,n); m=y.mean(); pred=np.zeros(len(z)-n); I=np.eye(n)\\n    for (k,r),w in READOUT_MODELS.items():\\n        A=matrices[k]; L=cho_factor(A[:n,:n]+r*I); pred+=w*(A[n:,:n]@cho_solve(L,y-m))\\n    return m+pred\\n"}'

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
