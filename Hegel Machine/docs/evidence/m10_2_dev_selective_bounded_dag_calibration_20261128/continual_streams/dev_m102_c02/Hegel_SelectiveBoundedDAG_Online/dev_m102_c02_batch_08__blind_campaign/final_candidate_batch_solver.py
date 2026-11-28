"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "2ea78d336eefbd2a7f0647f65e9c7c3f76fc0967155ff6cc8a57bdff9cc7f950"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated operator features for the scalar and temporal readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import root, minimize_scalar\\nfrom scipy.linalg import solve\\n\\n\\ndef _lin(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=1e-9)[0]\\n\\n\\ndef _cal(p, points):\\n    u, v = (np.asarray(p[k], float) for k in (\'u\', \'v\'))\\n    w = np.roll(u, -1, axis=1)\\n    d = np.c_[_lin(u), u*w, u**3]\\n    error = np.linalg.norm(v-d@_ls(d,v),axis=1)\\n    keep = np.argsort(error)[:-3]\\n    coeff = np.array([_ls(np.c_[np.ones(len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3][keep],v[keep,j]) for j in range(4)])\\n    def forward(z):\\n        w = np.roll(z,-1)\\n        return coeff[:,0]+coeff[:,1]*z+coeff[:,2]*w+coeff[:,3]*z*w+coeff[:,4]*z**3\\n    return np.array([root(lambda z:forward(z)-x,x).x for x in points])\\n\\n\\ndef _op(p,k,z,mode):\\n    u,v = np.asarray(p[\'u\'],float),np.asarray(p[k],float)\\n    def design(x,a):\\n        if mode==\'b\':\\n            return np.c_[_lin(x/(1+a*x*x)),x]\\n        return np.c_[_lin(np.arctan(a*x)),x,x**3]\\n    def error(a):\\n        d=design(u,a)\\n        h=np.sum(np.linalg.qr(d,mode=\'reduced\')[0]**2,axis=1)\\n        return np.mean(np.abs((v-d@_ls(d,v))/(1-h[:,None])))\\n    a=minimize_scalar(error,bounds=(.15,4),method=\'bounded\').x\\n    return design(z,a)@_ls(design(u,a),v)\\n\\n\\ndef _cross(a,b):\\n    return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)\\n\\n\\ndef _ridge(a,y,q,r):\\n    scale=np.maximum(a.std(0),1e-4)\\n    scale[0]=1\\n    p=np.eye(a.shape[1]);p[0,0]=0\\n    return (q/scale)@_ls(np.r_[a/scale,r*p],np.r_[y,np.zeros((a.shape[1],)+y.shape[1:])])\\n\\n\\ndef _kernel(f, b, y, n, ell, reg):\\n    b=b/np.maximum(b[:n].std(0),1e-8)\\n    k=np.exp(-np.mean((b[:,None]-b[None,:n])**2,axis=2)/(2*ell**2))\\n    a=f[:n]; m=a.shape[1]\\n    g=solve(k[:n]+reg*np.eye(n),np.c_[a,y],assume_a=\'pos\')\\n    beta=_ls(a.T@g[:,:m],a.T@g[:,m:])\\n    return f[n:]@beta+k[n:]@(g[:,m:]-g[:,:m]@beta)\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    tr=study[\'train\']; x=np.asarray(tr[\'x\'],float); q=np.asarray(study[\'test\'][\'x\'],float); n=len(x)\\n    z=_cal(study[\'panel_a\'],np.r_[x,q]); b=_op(study[\'panel_b\'],\'v\',z,\'b\'); c=_op(study[\'panel_c\'],\'v1\',z,\'c\')\\n    f=_lin(np.c_[b,c,_cross(b,c)])\\n    y=np.asarray(tr[\'y\'],float)[:,None]\\n    y_hat=(.75*_kernel(f,b,y,n,2.,.001)+.25*_kernel(f,b,y,n,1.,.03)).ravel()\\n    sf=_lin(np.c_[z,b,_cross(z,b)])\\n    state=_ridge(sf[:n],np.asarray(tr[\'traj\'],float),sf[n:],.08)\\n    # The terminal state carries much of the scalar readout\'s interaction.\\n    # Fit the difference on measured training states, then add it to the\\n    # predicted terminal state. Keep the independent readout as a hedge.\\n    diff = y[:,0] - np.asarray(tr[\'traj\'], float)[:,-1]\\n    z5 = z**5\\n    gf = _lin(np.c_[b, z5, _cross(b,z5)])\\n    u = np.asarray(study[\'panel_c\'][\'u\'],float)\\n    poly = lambda x: np.c_[_lin(x),x*x,x**3]\\n    d = poly(z) @ _ls(poly(u),np.asarray(study[\'panel_c\'][\'v2\'],float))\\n    df = _lin(np.c_[b,d])\\n    correction = (.65*_ridge(gf[:n],diff,gf[n:],.12)\\n                  +.35*_ridge(df[:n],diff,df[n:],.4))\\n    y_hat = .40*y_hat + .60*(state[:,-1]+correction)\\n    return {\'y_hat\':y_hat,\'state_pred\':state,\'diagnostics_vec\':np.array([y_hat.mean(),y_hat.std(),state[:,-1].mean(),state[:,-1].std()])}\\n"}'

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
