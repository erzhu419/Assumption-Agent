"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_11", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "dac3b2e410ecf8e33a47ffc37c1ffa3cd2e8728332cbd3b9562f51dd3462e7c2", "node_01": "685c1ae751dc557fed5b90d10fde021fe8d62da162c727c724ba09b8ea208ac2", "node_02": "6b2aed9db8bed4ef4803a05cc4d3b369a6ae5906b5a63da862248cde58bac85e"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Reference-panel calibration and spatial feature blocks for the frozen fitter.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = (np.asarray(panel[key], float) for key in (\'u\', \'v\'))\\n    good = np.linalg.norm(v, axis=1) < 5\\n    u, v = u[good], v[good]\\n    def design(z, j, h):\\n        a, b = z[:, j], z[:, (j+2)%4]\\n        return np.c_[np.ones(len(z)), a, b, a**3/(1+h*a*a), b**3/(1+h*b*b)]\\n    def profile(h, coeff=False):\\n        d = [design(u,j,h[0]) for j in range(4)]\\n        c = [_fit(a,v[:,j]) for j,a in enumerate(d)]\\n        return c if coeff else np.column_stack([a@q for a,q in zip(d,c)]).ravel()-v.ravel()\\n    h = least_squares(profile, [.48], bounds=(.02,3.)).x\\n    c = profile(h, True)\\n    def forward(z):\\n        return np.array([design(z[None],j,h[0])[0]@c[j] for j in range(4)])\\n    return np.array([least_squares(lambda z: forward(z)-row, row, gtol=1e-11).x for row in x])\\n\\n\\ndef _bdesign(u, h, kind):\\n    a,b=h\\n    if kind==\'sin\': return np.c_[np.ones(len(u)),u,np.sin(a*u)]\\n    if kind==\'sintanh\': return np.c_[np.ones(len(u)),u,np.sin(a*u),np.tanh(b*u)]\\n    if kind==\'sinatan\': return np.c_[np.ones(len(u)),u,np.sin(a*u),np.arctan(b*u)]\\n    if kind==\'sincos\': return np.c_[np.ones(len(u)),u,np.sin(a*u),np.cos(b*u)-1]\\n    if kind==\'warp\': return np.c_[np.ones(len(u)),u+b*np.sin(a*u)]\\n    if kind==\'rational\': return np.c_[np.ones(len(u)),u+b*np.arctan(a*u)]\\n    return np.c_[np.ones(len(u)),u]\\n\\n\\ndef _branch(panel,z,kind=\'sin\',report=False):\\n    u,v=(np.asarray(panel[key],float) for key in (\'u\',\'v\'))\\n    def residual(h):\\n        a=_bdesign(u,h,kind); return (a@_fit(a,v)-v).ravel()\\n    bounds=([.3,-3],[9,3]) if kind in (\'warp\',\'rational\') else ([.3,.3],[9,9])\\n    o=min((least_squares(residual,[a,b],bounds=bounds,max_nfev=100) for a in (1.,2.9,5.5,8.) for b in ((-.4,.4) if kind in (\'warp\',\'rational\') else (1.,3.))),key=lambda o:o.cost)\\n    h=o.x\\n    if report: print(\'B\',kind,h,np.sqrt(np.mean(o.fun**2)))\\n    return _bdesign(z,h,kind)@_fit(_bdesign(u,h,kind),v), h\\n\\n\\ndef _quad(x):\\n    i,j=np.triu_indices(x.shape[1]);return x[:,i]*x[:,j]\\n\\n\\ndef _blocks(z,b,h):\\n    o=np.ones(len(z))\\n    R=np.c_[z,_quad(z),o,_quad(b),_quad(np.c_[z,b])[:,10:26],b]\\n    S=np.c_[z,o,_quad(z),_quad(b),b]\\n    return R,S\\n\\n\\n\\ndef _readout_dict(s,z,b,h):\\n    f=h[0]; H=np.sin(f*z); W=z-.4*H\\n    out={\'z\':z,\'b\':b,\'h\':H,\'w\':W,\'wa\':z-.30*H,\'wb\':z-.50*H,\\n         \'t\':np.tanh(1.3*z),\'q\':z/(1+1.3*z*z),\'e\':np.cos(f*z)-1}\\n    for mode,name in [(\'rational\',\'br\'),(\'warp\',\'bw\'),(\'sintanh\',\'bt\'),(\'sinatan\',\'ba\')]:\\n        if name in (\'z\', \'ba\', \'ba\'):\\n            out[name]=_branch(s[\'panel_b\'],z,mode)[0]\\n    return out\\n\\n\\ndef _readout_form(d, spec):\\n    a,f,g,mode=spec; A,F,G=d[a],d[f],d[g]; n=len(A); one=np.ones(n)\\n    extra=[one]\\n    if mode in (\'quadratic\',\'mixed\',\'cube\',\'rational\',\'harmonic\',\'extended\'):\\n        extra.append(_quad(G))\\n    if mode in (\'mixed\',\'cube\',\'extended\'):\\n        extra.append((A[:,:,None]*G[:,None,:]).reshape(n,-1))\\n    if mode in (\'add\',\'cube\',\'extended\'):\\n        extra.extend((A**3,G**3))\\n    if mode==\'rational\':\\n        extra.extend((A/(1+A*A),G/(1+G*G),np.tanh(G)))\\n    if mode==\'harmonic\':\\n        extra.extend((np.sin(2.85*A),np.cos(2.85*A)-1,np.sin(2.85*G),np.cos(2.85*G)-1))\\n    if mode==\'extended\':\\n        extra.extend((A**2*G,A*G**2))\\n    return np.c_[A,_quad(F),*extra,G]\\ndef run_node(inputs: dict) -> dict:\\n    s=inputs[\'__input__\']; x=s[\'train\'][\'x\']; n=len(x)\\n    z=_calibrate(s[\'panel_a\'],np.r_[x,s[\'test\'][\'x\']]); b,h=_branch(s[\'panel_b\'],z)\\n    _,t=_blocks(z,b,h)\\n    r=_readout_form(_readout_dict(s,z,b,h), (\'z\', \'ba\', \'ba\', \'harmonic\'))\\n    return {\'readout_train\':r[:n],\'readout_test\':r[n:],\'z\':t[:n],\'zt\':t[n:]}\\n", "node_01": "\\"\\"\\"Fit spatial readout and shared trajectory coefficients.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef _fit(x,y,blocks):\\n    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-8)\\n    A=x/scale\\n    p=A.shape[1]; G=A.T@A; B=A.T@y\\n    best=None\\n    for penalty in blocks:\\n        for lam in (.002,.02,.1,.5,2.,8.):\\n            gram=G+np.diag(lam*penalty)\\n            inv=np.linalg.solve(gram,A.T)\\n            c=inv@y\\n            r=(A@c-y)/(np.maximum(1.-np.sum(A*inv.T,axis=1),1e-7).reshape((-1,)+(1,)*(y.ndim-1)))\\n            val=np.mean(abs(r))\\n            if best is None or val<best[0]: best=(val,c)\\n    return best[1]/scale.reshape((-1,)+(1,)*(y.ndim-1))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    raw=inputs[\'__input__\']; f=inputs[\'node_00\']; A=f[\'readout_train\']; B=f[\'z\']\\n    y=np.asarray(raw[\'train\'][\'y\']); t=np.asarray(raw[\'train\'][\'traj\']); _,_,q=np.linalg.svd(t,full_matrices=False); q=q[:2]\\n    ry=[np.r_[np.ones(4), np.full(10,k), np.full(A.shape[1]-18,j), np.full(4,j*3)] for k in (.15,1.,10.) for j in (3.,30.,300.)]\\n    rs=[np.r_[np.ones(4),np.full(B.shape[1]-8,j),np.full(4,j*k)] for j in (3.,30.,300.) for k in (.3,3.,30.)]\\n    return {\'readout_coef\':_fit(A,y,ry),\'state_coef\':_fit(B,t@q.T,rs)@q}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    params = inputs[\'node_01\']\\n    yh = features[\'readout_test\'] @ params[\'readout_coef\']\\n    sp = features[\'zt\'] @ params[\'state_coef\']\\n    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])\\n    return {\'y_hat\': yh, \'state_pred\': sp, \'diagnostics_vec\': diag}\\n"}'

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
