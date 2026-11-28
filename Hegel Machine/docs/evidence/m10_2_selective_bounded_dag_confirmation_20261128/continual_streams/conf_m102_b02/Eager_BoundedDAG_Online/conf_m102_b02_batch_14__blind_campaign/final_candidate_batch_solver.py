"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_20", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "3808b4391c0b66ec23c82b65262d254949ab7bd027701c1e6407482660c82016", "node_01": "40d18571ae358b875c90f50f4c59694ae6a468cd9a5f81f7146e9452b72deeb0", "node_02": "d94d6bbf564efb7d9f3cfe9cd7fc376b592b0c235d5a682ced6b560fe99778fb"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\nfrom scipy.special import erf\\n\\n\\ndef _design(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=1e-9)[0]\\n\\n\\ndef _panel_a(study, x, verbose=False):\\n    p = study[\'panel_a\']\\n    u, v = p[\'u\'], p[\'v\']\\n    keep = np.linalg.norm(u-v, axis=1) < 3\\n    u,v = u[keep],v[keep]\\n    # Diagonal and opposite-coordinate couplings share frequencies across axes.\\n    def parts(z, q, j):\\n        h = z[:, (j+2)%4]\\n        return _design(np.column_stack((z[:,j], np.sin(q[0]*z[:,j]), h, np.sin(q[1]*h))))\\n    def err(q):\\n        return np.concatenate([F@_fit(F,v[:,j])-v[:,j] for j in range(4) for F in [parts(u,q,j)]])\\n    opt = least_squares(err,[2.45,2.45],bounds=([.5,.5],[4,4]))\\n    q=opt.x\\n    co=np.stack([_fit(parts(u,q,j),v[:,j]) for j in range(4)])\\n    def forward(z):\\n        return np.column_stack([parts(z,q,j)@co[j] for j in range(4)])\\n    z=x-co[:,0]\\n    for _ in range(35):\\n        r=forward(z)-x\\n        J=np.zeros((len(z),4,4))\\n        for j in range(4):\\n            k=(j+2)%4\\n            J[:,j,j]=co[j,1]+co[j,2]*q[0]*np.cos(q[0]*z[:,j])\\n            J[:,j,k]=co[j,3]+co[j,4]*q[1]*np.cos(q[1]*z[:,k])\\n        d=np.linalg.solve(J, r[:,:,None])[:,:,0]\\n        z-=np.clip(d,-.2,.2)\\n    if verbose: print(\'panel A\',q, np.sqrt(np.mean(opt.fun**2)),np.max(abs(forward(z)-x)),co)\\n    return z\\n\\n\\ndef _panel_b(study,z):\\n    p=study[\'panel_b\']; u,v=p[\'u\'],p[\'v\']\\n    def residual(q):\\n        f=_design(np.cos(q*u)); return np.mean((f@_fit(f,v)-v)**2)\\n    grid=np.linspace(.4,7,23); i=int(np.argmin([residual(q) for q in grid]))\\n    q=minimize_scalar(residual,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method=\'bounded\').x\\n    return _design(np.cos(q*z))@_fit(_design(np.cos(q*u)),v)\\n\\n\\n\\n\\ndef _panel_c(study):\\n    p = study[\'panel_c\']; u = p[\'u\']; i,j = np.triu_indices(4,1)\\n    def train(y, func, q0, inter):\\n        def design(x,q):\\n            return _design(np.column_stack((x,func(q*x), *([x[:,i]*x[:,j]] if inter else [x*x]))))\\n        def err(q):\\n            f=design(u,q); return np.mean((f@_fit(f,y)-y)**2)\\n        q=minimize_scalar(err,bounds=q0,method=\'bounded\').x\\n        c=_fit(design(u,q),y)\\n        return lambda x: design(x,q)@c\\n    return train(p[\'v1\'],np.tanh,(.7,2.7),True), train(p[\'v2\'],np.cos,(2,3),False)\\n\\n\\ndef _estimate(s, debug=False):\\n    from scipy.linalg import cho_factor, cho_solve\\n    tr=s[\'train\']; n=len(tr[\'x\']); x=np.vstack((tr[\'x\'],s[\'test\'][\'x\']))\\n    z=_panel_a(s,x); b=_panel_b(s,z); h,g=_panel_c(s)\\n    traj=np.asarray(tr[\'traj\']); U=np.linalg.svd(traj,full_matrices=False)[2][:2]; y=np.column_stack((tr[\'y\'],traj@U.T)); sy=np.maximum(y.std(0),1e-7); Y=(y-y.mean(0))/sy\\n    ids=np.eye(n); ix,jx=np.triu_indices(4,1); records=[]; eye=np.eye(n)\\n    def normalize(f):\\n        f=f-f[:n].mean(0); return f/np.maximum(f[:n].std(0),1e-5)\\n    def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)\\n    def evaluate(F,name,types=(0,1,2),lams=(.000006,.0004,.008)):\\n        F=normalize(F); L=(F@F[:n].T)/F.shape[1]\\n        for typ in types:\\n            if typ==0: K=1+L\\n            elif typ==1: K=(1+L)**2\\n            elif typ==2: K=(1+L)**3\\n            elif typ==3:\\n                norms=np.sum(F*F,axis=1)/F.shape[1]; d=np.maximum(norms[:,None]+norms[None,:n]-2*L,0);K=(1+L)**2+np.exp(-.6*d)\\n            for lam in lams:\\n                try: C=cho_factor(K[:n]+lam*eye,check_finite=False); inv=cho_solve(C,ids,check_finite=False); alpha=inv@Y\\n                except (ValueError,np.linalg.LinAlgError):continue\\n                residual=alpha/np.diag(inv)[:,None]; error=np.mean(abs(residual),axis=0)\\n                records.append((error, name, typ, lam, K[n:]@alpha, Y-lam*alpha, residual))\\n    base={\'z\':z,\'b\':b,\'x\':x,\'hz\':h(z),\'gz\':g(z),\'hb\':h(b),\'gb\':g(b)}\\n    for w in (-1,-.6,-.3,-.15,.15,.3,.6,1):\\n        a=b+w*z; k=f\'b{w:+g}z\';base[k]=a\\n        if abs(w) in (.3,.6,1): base[\'h\'+k]=h(a);base[\'g\'+k]=g(a)\\n    for w in (-1,-.5,.5,1):\\n        base[f\'hb{w:+g}gb\']=base[\'hb\']+w*base[\'gb\'];base[f\'b{w:+g}hb\']=b+w*base[\'hb\'];base[f\'b{w:+g}gb\']=b+w*base[\'gb\']\\n    stats=np.asarray(s.get(\'telemetry_vec\',[2,2,2,2])); qs=np.unique(np.r_[.5,.8,1.1,1.4,1.7,2,2.4,2.6,3.1,3.6,4.2,5.,6.,stats[(stats>.4)&(stats<7)]])\\n    forms={\'sin\':np.sin,\'cos\':np.cos,\'tanh\':np.tanh,\'erf\':erf,\'rat\':lambda x:x/(1+x*x),\'soft\':lambda x:x/np.sqrt(1+x*x)}\\n    for name,a in base.items():\\n        evaluate(a,name,(0,1,2,3))\\n        evaluate(np.c_[a,a**2,a**3,a[:,ix]*a[:,jx]],name+\'/poly\',(0,1))\\n        evaluate(np.c_[a,a**2,a**3,a**4,a[:,ix]*a[:,jx],a[:,ix]**2*a[:,jx],a[:,ix]*a[:,jx]**2],name+\'/tensor\',(0,))\\n        for fn,f in forms.items():\\n            for q in qs:\\n                w=f(q*a); tag=f\'{name}/{fn}/{q:g}\'\\n                evaluate(np.c_[a,w],tag,(0,1,2),(.00003,.003))\\n                evaluate(np.c_[a,a*a,w,w*w,cross(a,w)],tag+\'/mixed\',(0,),(.000008,.0007))\\n                evaluate(w,tag+\'/only\',(0,1),(.00004,.002))\\n    pairs=[(\'b\',\'z\'),(\'b\',\'hb\'),(\'b\',\'gb\'),(\'z\',\'hb\'),(\'z\',\'gb\'),(\'hb\',\'gb\'),(\'hz\',\'gz\'),(\'b\',\'hz\'),(\'b\',\'gz\'),(\'z\',\'hz\'),(\'z\',\'gz\')]\\n    for w in (.15,.3,.6,1):\\n        for prefix in (\'\',\'h\',\'g\'):\\n            k=f\'{prefix}b{w:+g}z\'; l=f\'{prefix}b{-w:+g}z\';\\n            if k not in base:continue\\n            pairs.extend(((k,l),(\'b\',k),(\'z\',k)))\\n    for ka,kb in pairs:\\n        a,c=base[ka],base[kb]; name=ka+\'|\'+kb\\n        evaluate(np.c_[a,c],name,(0,1,2,3))\\n        evaluate(np.c_[a,c,a*a,c*c,cross(a,c)],name+\'/mixed\',(0,1,2))\\n        for q in (.8,1.4,2.15,2.6,3.1,4,5):\\n            for fn,f in forms.items():\\n                w=f(q*a); v=f(q*c); tag=f\'{name}/{fn}/{q:g}\'\\n                evaluate(np.c_[a,c,w,v],tag,(0,1,2),(.00004,.003))\\n                evaluate(np.c_[a,c,w,v,a*a,c*c,cross(w,v)],tag+\'/interaction\',(0,),(.00004,.003))\\n                evaluate(np.c_[w,v,cross(w,v)],tag+\'/pure\',(0,1),(.00004,.003))\\n    err=np.array([r[0] for r in records]); yp=[]; tp=[]\\n    for j in range(Y.shape[1]):\\n        best=np.argsort(err[:,j]); s0=err[best[0],j]; chosen=[]; seen=set()\\n        for k in best:\\n            e,name,typ,lam,test,train,resid=records[k]\\n            if e[j]>s0*1.3 or len(chosen)>=18:break\\n            # Do not let nearby frequency or ridge variants dominate the ensemble.\\n            family=\'/\'.join(v for v in name.split(\'/\') if not v.replace(\'.\',\'\').isdigit())\\n            if family in seen:continue\\n            seen.add(family); chosen.append(k)\\n        wt=np.exp(-9*(err[chosen,j]/max(s0,1e-9)-1));wt/=wt.sum()\\n        tp.append(sum(w*records[k][4][:,j] for k,w in zip(chosen,wt)))\\n        yp.append(sum(w*records[k][5][:,j] for k,w in zip(chosen,wt)))\\n        if debug: print(\'mode\',j,\'PRESS\',s0*sy[j],[(records[k][1:4],round(err[k,j]*sy[j],5)) for k in best[:7]],\'ensemble\',len(chosen),flush=True)\\n    return np.vstack((np.column_stack(yp),np.column_stack(tp))), np.c_[z,b], n\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study=inputs[\'__input__\']; p,a,n=_estimate(study)\\n    f=np.column_stack((1.35*p,.075*a));\\n    return {\'train_features\':f[:n], \'test_features\':f[n:]}\\n", "node_01": "import numpy as np\\n\\n\\ndef _kernel(x, w):\\n    linear = x @ w.T / x.shape[1]\\n    r = .5 * np.abs(x[:, None, :] - w[None, :, :])\\n    g = (1 + r + r*r/3)*np.exp(-r)\\n    avg = g.mean(2)\\n    pair = avg*avg - (g*g).sum(2)/x.shape[1]**2\\n    return (1+linear)**2 + .30*avg + .30*pair\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    i, j = features[\'train_features\'], features[\'test_features\']\\n    k, ktest = _kernel(i, i), _kernel(j, i)\\n    return {\'k\': k, \'ktest\': ktest}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    tr = inputs[\'__input__\'][\'train\']\\n    kernels = inputs[\'node_01\']\\n    k, ktest = kernels[\'k\'], kernels[\'ktest\']\\n    n = k.shape[0]\\n    traj = np.asarray(tr[\'traj\'])\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    y = ktest @ np.linalg.solve(k + .004*np.eye(n), tr[\'y\'])\\n    state = (ktest @ np.linalg.solve(k + .004*np.eye(n), traj @ modes.T)) @ modes\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
