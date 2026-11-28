"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "8f446ed7ad6bffcfcbb184adfc880d3f711db7b243ce75710f738c2b9cd8c120"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated regression with separate readout and temporal response kernels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize\\nfrom scipy.linalg import cho_factor, cho_solve\\n\\n\\ndef fit(a,b):\\n    return np.linalg.lstsq(a,b,rcond=1e-11)[0]\\n\\n\\ndef affine(x):\\n    return np.c_[np.ones(len(x)),x]\\n\\n\\ndef calibrate(panel,x):\\n    u,v=[np.asarray(panel[k],float) for k in (\'u\',\'v\')]\\n    good=np.linalg.norm(v,axis=1)<5\\n    def design(u,j):\\n        a=u[:,j]; b=u[:,(j+1)%4]\\n        return np.c_[affine(u),a*b,a**3]\\n    co=np.column_stack([fit(design(u[good],j),v[good,j]) for j in range(4)])\\n    def forward(z):\\n        return np.column_stack([design(z,j)@co[:,j] for j in range(4)])\\n    g=np.linalg.solve(co[1:5].T,(x-co[0]).T).T\\n    return np.array([least_squares(lambda z:forward(z[None])[0]-r,a,gtol=1e-12).x for a,r in zip(g,x)])\\n\\n\\n\\n\\ndef _panel_b_warp(panel):\\n    \\"\\"\\"Profile out the affine mixing to estimate a shared odd coordinate warp.\\"\\"\\"\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n\\n    def warp(x, h):\\n        a,b,c,d = h\\n        return x+a*np.arctan(b*x)+d*x**3/(1+c*x*x)\\n\\n    def residual(h):\\n        A = affine(warp(u,h))\\n        return (A@fit(A,v)-v).ravel()\\n\\n    starts = [(-.45,2.15,12.,-3.3),(-.5,2.,8.,-2.),(-1.,2.,1.,.2),(-.6,1.5,2.,0.)]\\n    opt = min((least_squares(residual, h, bounds=([-8,0,0,-8],[8,10,16,8]), max_nfev=200) for h in starts), key=lambda r:r.cost)\\n    return lambda x:warp(x,opt.x)\\n\\n\\ndef _standard(x,n):\\n    return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-8)\\n\\n\\ndef _components(z,b,n):\\n    a,x = _standard(z,n),_standard(b,n)\\n    A,G = (q@q.T/q.shape[1] for q in (a,x))\\n    R = ((x[:,None]-x[None,:])**2+.25*(a[:,None]-a[None,:])**2)/1.25\\n    K=[np.ones_like(A),A,A*A,G,G*G,G*A]\\n    for L in (1.5,6,24):\\n        S=np.exp(-R/L)\\n        M=S.sum(2)\\n        K.extend([M/4,(M*M-(S*S).sum(2))/12])\\n        K.extend(S[:,:,i] for i in range(4))\\n        K.extend(S[:,:,i]*S[:,:,j] for i in range(4) for j in range(i))\\n    return np.array(K)\\n\\n\\ndef _profile_kernel(B,Y,n,penalty=.02):\\n    Y=np.asarray(Y).reshape(n,-1)\\n    mu=Y.mean(0)\\n    sd=np.maximum(Y.std(0),1e-8)\\n    M=(Y-mu)/sd\\n    eye=np.eye(n)\\n    p=len(B)\\n    A=np.concatenate([B[:,:n,:n],eye[None]],axis=0).reshape(p+1,-1)\\n    ref=np.r_[np.full(p,-3.),-7.]\\n    ref[0]=-1\\n\\n    def objective(q):\\n        v=np.exp(q)\\n        F=cho_factor((v@A).reshape(n,n),check_finite=False)\\n        alpha=cho_solve(F,M,check_finite=False)\\n        D=(cho_solve(F,eye,check_finite=False)-alpha@alpha.T/M.shape[1]).ravel()\\n        d=q-ref\\n        val=np.log(F[0].diagonal()).sum()+np.sum(M*alpha)/(2*M.shape[1])+penalty*(d@d)/2\\n        return val,(A@D)*v/2+penalty*d\\n\\n    O=minimize(objective,ref,jac=True,method=\'L-BFGS-B\',bounds=[(-14,5)]*p+[(-12,-2)],options={\'maxiter\':160,\'ftol\':1e-7})\\n    v=np.exp(O.x)\\n    F=cho_factor((v@A).reshape(n,n),check_finite=False)\\n    W=(v[:-1]@B[:,n:,:n].reshape(p,-1)).reshape(-1,n)\\n    return (W@cho_solve(F,M,check_finite=False))*sd+mu\\n\\n\\ndef solve_batch(study):\\n    tr=study[\'train\']\\n    n=len(tr[\'x\'])\\n    z=calibrate(study[\'panel_a\'],np.r_[tr[\'x\'],study[\'test\'][\'x\']])\\n    b=_panel_b_warp(study[\'panel_b\'])(z)\\n    B=_components(z,b,n)\\n    yh=_profile_kernel(B,tr[\'y\'],n)[:,0]\\n    a=_standard(z,n)\\n    w=_standard(z-.4*np.sin(2.9*z),n)\\n    K=(1+(w@w.T+.1*a@a.T)/4.4)**2\\n    T=np.asarray(tr[\'traj\'],float)\\n    m=T.mean(0)\\n    baseline=K[n:,:n]@cho_solve(cho_factor(K[:n,:n]+.001*np.eye(n)),T-m)+m\\n    P=.5*baseline+.5*_profile_kernel(B,T,n)\\n    last=P[:,-1]\\n    return {\'y_hat\':yh,\'state_pred\':P,\'diagnostics_vec\':np.array([yh.mean(),yh.std(),last.mean(),last.std()])}\\n"}'

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
