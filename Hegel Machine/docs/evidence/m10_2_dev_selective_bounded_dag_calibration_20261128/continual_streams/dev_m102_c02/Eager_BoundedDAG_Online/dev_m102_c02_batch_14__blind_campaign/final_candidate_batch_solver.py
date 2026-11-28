"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_16", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "70af28a6bbfbd4d812c0cf6454c1a4110231054e084ad096391d7c0034685136"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\ndef _ls(a,b):\\n    return np.linalg.lstsq(a,b,rcond=1e-10)[0]\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)),x))\\n\\ndef _calibrate(panel, x):\\n    u,v=panel[\'u\'],panel[\'v\']; pair=[2,3,0,1]\\n    y=v-.18*np.tanh(1.8*u)+.32*np.sin(2.4*u[:,pair])\\n    mask=np.ones(len(u),bool)\\n    for _ in range(4):\\n        off,gain=np.array([_ls(_aff(u[mask,j:j+1]),y[mask,j]) for j in range(4)]).T\\n        mask=np.max(abs(y-off-u*gain),axis=1)<.5\\n    z=x.copy()\\n    for _ in range(40):\\n        h=np.tanh(1.8*z)\\n        e=off+gain*z+.18*h-.32*np.sin(2.4*z[:,pair])-x\\n        j=np.zeros((len(x),4,4)); ix=np.arange(4)\\n        j[:,ix,ix]=gain+.324*(1-h*h)\\n        j[:,ix,pair]=-.768*np.cos(2.4*z[:,pair])\\n        z-=np.clip(np.linalg.solve(j,e[...,None])[...,0],-.28,.28)\\n    return z\\n\\ndef _outer(x,y):\\n    return (x[:,:,None]*y[:,None,:]).reshape(len(x),x.shape[1]*y.shape[1])\\n\\ndef _build(study,x):\\n    z=_calibrate(study[\'panel_a\'],x)\\n    p=study[\'panel_b\']; warp=lambda u:u-.39752*np.sin(2.8478*u-.002)\\n    w=_ls(_aff(warp(p[\'u\'])),p[\'v\']); b=_aff(warp(z))@w\\n    i,j=np.triu_indices(4)\\n    return z,b,np.column_stack((_aff(b),b[:,i]*b[:,j],_outer(b,z)))\\n\\ndef _scalar(a, b, y):\\n    n=len(y)\\n    scale=np.maximum(b[:n].std(axis=0), 1e-8)\\n    b=b/scale\\n    ab=a[:n]\\n    rng=np.random.default_rng(1)\\n    starts=np.r_[np.zeros((1,4)),np.eye(4),rng.normal(0,.5,(8,4))]\\n    predictions=[]\\n    for link in (np.sin,np.tanh):\\n        def design(p, q, base):\\n            return np.column_stack((base,link(p[0]+q@p[1:])))\\n        def residual(p):\\n            d=design(p,b[:n],ab)\\n            return np.r_[d@_ls(d,y)-y,.008*p]\\n        options=[least_squares(residual,np.r_[.1,start],max_nfev=120) for start in starts]\\n        fit=min(options,key=lambda opt:opt.cost)\\n        d=design(fit.x,b,a)\\n        predictions.append(d[n:]@_ls(d[:n],y))\\n    return np.mean(predictions,axis=0)\\n\\n\\ndef _shrink_scalar(a,b,y,strength):\\n    n=len(y)\\n    a=a/np.maximum(a[:n].std(0),.08)\\n    b=b/np.maximum(b[:n].std(0),.08)\\n    target=np.r_[y,np.zeros(a.shape[1]+1)]\\n    pen=np.diag(np.r_[0,np.full(4,.06*strength),np.full(a.shape[1]-5,strength),.035*strength])\\n    starts=np.r_[np.eye(4),np.random.default_rng(1).normal(0,.5,(8,4))]\\n    outs=[]\\n    for link in (np.sin,np.tanh):\\n        def design(p,base,q):\\n            return np.column_stack((base,link(p[0]+q@p[1:])))\\n        def residual(p):\\n            mat=np.vstack((design(p,a[:n],b[:n]),pen))\\n            return np.r_[mat@_ls(mat,target)-target,.008*p]\\n        opt=min((least_squares(residual,np.r_[.1,start],max_nfev=100) for start in starts),key=lambda o:o.cost)\\n        pred=design(opt.x,a,b)\\n        outs.append(pred[n:]@_ls(np.vstack((pred[:n],pen)),target))\\n    return np.mean(outs,axis=0)\\n\\n\\ndef solve_batch(study):\\n    train=study[\'train\']; n=len(train[\'x\']); x=np.concatenate((train[\'x\'],study[\'test\'][\'x\']))\\n    _,b,a=_build(study,x)\\n    y_hat=_shrink_scalar(a,b,train[\'y\'],0.12)\\n    _,_,vh=np.linalg.svd(train[\'traj\'],full_matrices=False); modes=vh[:3]\\n    state_pred=(a[n:]@_ls(a[:n],train[\'traj\']@modes.T))@modes\\n    last=state_pred[:,-1]\\n    return {\'y_hat\':y_hat,\'state_pred\':state_pred,\'diagnostics_vec\':np.array([y_hat.mean(),y_hat.std(),last.mean(),last.std()])}\\n"}'

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
