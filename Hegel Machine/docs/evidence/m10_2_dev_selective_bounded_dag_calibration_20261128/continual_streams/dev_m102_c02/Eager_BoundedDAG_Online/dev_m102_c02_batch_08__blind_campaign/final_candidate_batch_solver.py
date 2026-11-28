"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 4, "fanout_by_node": {"node_00": 1, "node_01": 2, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_01"], "node_03": ["node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "c921cf53600f8bf1f6d7ab5cbd731e847fbaa7b64b86edc3cfcd0a169417e860", "node_01": "33cd376c2c5348e1cd75ec198c5e38d2f8b82db3e98669cbbc348c4199303eea", "node_02": "f9d37d1da43c0fe301660ed1cdcf195ba0d697d8fe1431e78060b5e508e99eaf", "node_03": "b6748aa005c40c6d864a932b98a4c28a1dc3a31eb8ffcc85dab661ae26874a4c"}, "shared_fanout_nodes": ["node_01"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates from the paired calibration panels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _a_basis(z):\\n    w = np.roll(z, -1, axis=-1)\\n    return np.stack((np.ones_like(z), z, w, z*w, z**3), axis=-1)\\n\\n\\ndef _sensor_inverse(panel, x):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\',\'v\'))\\n    a = _a_basis(u).transpose(1,0,2)\\n    rng = np.random.default_rng(2)\\n    ids = np.argsort(rng.random((96,len(u))),axis=1)[:,:6]\\n    coeff = np.linalg.pinv(a[:,ids]) @ v.T[:,ids,None]\\n    err = np.sqrt(np.sum((np.einsum(\'dnp,dsp->dsn\',a,coeff[...,0])-v.T[:,None,:])**2,axis=0))\\n    count = max(6,int(.8*len(u)))\\n    loss = np.mean(np.partition(err,count-1,axis=1)[:,:count]**2,axis=1)\\n    r = err[np.argmin(loss)]\\n    keep = r <= max(8*np.median(r),1e-3)\\n    p = np.stack([np.linalg.lstsq(a[j,keep],v[keep,j],rcond=None)[0] for j in range(u.shape[1])])\\n    def forward(z):\\n        return np.einsum(\'dp,dp->d\',_a_basis(z),p)\\n    def jac(z):\\n        j = np.arange(len(z)); w = np.roll(z,-1)\\n        out = np.zeros((len(z),len(z)))\\n        out[j,j] = p[:,1]+p[:,3]*w+3*p[:,4]*z*z\\n        out[j,(j+1)%len(z)] = p[:,2]+p[:,3]*z\\n        return out\\n    return np.array([least_squares(lambda z:forward(z)-row,row,jac=jac,gtol=1e-12).x for row in x]).reshape(x.shape)\\n\\n\\ndef _fit(panel, key, fun, bounds):\\n    u,v = (np.asarray(panel[k],dtype=float) for k in (\'u\',key))\\n    def basis(x,s): return np.column_stack((np.ones(len(x)),x,fun(x,s)))\\n    def loss(s):\\n        a=basis(u,s); inv=np.linalg.pinv(a); r=v-a@(inv@v)\\n        return np.mean(np.abs(r/np.maximum(1-np.sum(a*inv.T,axis=1),1e-8)[:,None]))\\n    s=minimize_scalar(loss,bounds=bounds,method=\'bounded\').x\\n    p=np.linalg.lstsq(basis(u,s),v,rcond=None)[0]\\n    return lambda x:basis(x,s)@p\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study=inputs[\'__input__\']; x=np.asarray(study[\'train\'][\'x\'],dtype=float)\\n    z=_sensor_inverse(study[\'panel_a\'],np.concatenate((x,study[\'test\'][\'x\'])))\\n    b=_fit(study[\'panel_b\'],\'v\',lambda q,s:q**3/(1+s*q*q),(.01,2.))\\n    d=_fit(study[\'panel_c\'],\'v2\',lambda q,s:np.expm1(-s*q*q),(.02,1.))\\n    h=_fit(study[\'panel_c\'],\'v1\',lambda q,s:np.arctan(s*q),(.1,5.))\\n    c=_coordinates_from_calibration(z,(b,h,d),study[\'train\'],len(x))\\n    return {\'c\':c[:len(x)],\'query\':c[len(x):]}\\n\\n\\ndef _coordinates_from_calibration(z, panels, train, n):\\n    from scipy.linalg import cho_factor, cho_solve\\n    from scipy.optimize import minimize\\n    b, h, d = panels\\n    p, q, r = b(z), h(z), d(z)\\n    bases = [z, p, q, r, b(p), h(p), d(p), b(q), h(q), d(q), b(r), h(r), d(r)]\\n    cores = [z, p, q, r]\\n    blocks = list(bases)\\n    for i, v in enumerate(cores):\\n        for w in cores[i:]:\\n            blocks.extend((v*w, v/(1+w*w), w/(1+v*v)))\\n            prod = (v[:,:,None]*w[:,None,:]).reshape(len(z),-1)\\n            blocks.append(prod)\\n    for v in cores:\\n        blocks.extend((v**2, v**3, np.tanh(2*v), np.arctan(3*v)))\\n        ip = v @ v.T\\n        dist = np.maximum(np.diag(ip)[:,None]+np.diag(ip)[None,:]-2*ip, 0.)\\n        for rate in (0.4, 1.6, 5.):\\n            blocks.append(np.exp(-rate*dist))\\n    for v in (p+q, p+r, q+r, p-q, p-r, q-r, p*q, p*r, q*r, p+q*r, (p+q)/(1+r*r)):\\n        blocks.extend((b(v),h(v),d(v)))\\n    grams = []\\n    for x in blocks:\\n        if x.shape == (len(z),len(z)):\\n            m=x[:n,:n].mean(); grams.append(x/max(m,0.1))\\n        else:\\n            x=x-x[:n].mean(axis=0)\\n            x=x/np.maximum(x[:n].std(axis=0),1e-7)\\n            grams.append((x@x.T)/x.shape[1])\\n    k=np.asarray(grams)\\n    kt=np.ascontiguousarray(k[:,:n,:n]).reshape(len(k),-1)\\n    tr=np.asarray(train[\'traj\'],float)\\n    mean=tr.mean(0)\\n    _,_,vt=np.linalg.svd(tr-mean,full_matrices=False)\\n    y=np.column_stack((np.asarray(train[\'y\'],float), (tr-mean)@vt[:2].T))\\n    mu=y.mean(0); sd=np.maximum(y.std(0),1e-8); y=(y-mu)/sd\\n    eye=np.eye(n)\\n    results=[]\\n    for j in range(3):\\n        target=y[:,j]\\n        def evaluate(logw, predict=False):\\n            w=np.exp(logw)\\n            cov=(w[:-1]@kt).reshape(n,n)+(1e-9+w[-1])*eye+0.3\\n            fact=cho_factor(cov,check_finite=False)\\n            alpha=cho_solve(fact,target,check_finite=False)\\n            if predict:\\n                cross=np.einsum(\'i,ijk->jk\',w[:-1],k[:,:,:n])+0.3\\n                return cross@alpha\\n            ic=cho_solve(fact,eye,check_finite=False)\\n            change=ic-np.outer(alpha,alpha)\\n            # A weak weight penalty discourages unnecessary large components.\\n            obj=0.5*(target@alpha)+np.sum(np.log(np.diag(fact[0])))+0.04*w[:-1].sum()\\n            grad=0.5*np.r_[kt@change.ravel(),np.trace(change)]\\n            grad[:-1]+=0.04\\n            return obj, grad*w\\n        init=np.r_[np.full(len(k),-4.),-2.5]\\n        fit=minimize(evaluate,init,jac=True,method=\'L-BFGS-B\',bounds=[(-14.,3.)]*len(k)+[(-9.,1.)],options={\'maxiter\':165,\'ftol\':1e-8})\\n        results.append(evaluate(fit.x,True))\\n    c=np.column_stack((*results,np.zeros(len(z))))\\n    return c\\n", "node_01": "\\"\\"\\"Fit the scalar readout and pass the affine trajectory projection downstream.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _interaction(x):\\n    return np.sum(x[:, :2] * x[:, 2:], axis=1)\\n\\n\\ndef _readout(x, interaction, params):\\n    z = x @ params[1:5]\\n    scale, weight = params[6:8]\\n    return params[0] + params[5] * interaction + (1.0 - weight) * z + weight * np.arctan(scale * z) / scale\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coords = inputs[\'node_00\']\\n    c = np.asarray(coords[\'c\'], dtype=float)\\n    query = np.asarray(coords[\'query\'], dtype=float)\\n    train = study[\'train\']\\n    y = np.asarray(train[\'y\'], dtype=float)\\n    design = _affine(c)\\n    new_design = _affine(query)\\n    interaction = _interaction(c)\\n    initial = np.linalg.lstsq(np.column_stack((design, interaction)), y, rcond=None)[0]\\n    params = least_squares(\\n        lambda p: _readout(c, interaction, p) - y,\\n        np.r_[initial, 0.35, 1.0],\\n        bounds=(np.r_[np.full(6, -np.inf), 0.015, 0.0],\\n                np.r_[np.full(6, np.inf), 1.5, 5.0]),\\n        max_nfev=200,\\n    ).x\\n    beta = np.linalg.lstsq(design, np.asarray(train[\'traj\'], dtype=float), rcond=None)[0]\\n    times = np.asarray(study[\'eval_times\'], dtype=float)\\n    t = times / max(float(np.max(np.abs(times))), np.finfo(float).tiny)\\n    return {\'y_hat\': _readout(query, _interaction(query), params),\\n            \'fitted\': design @ beta, \'beta\': beta, \'new_design\': new_design, \'t\': t}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    stage = inputs[\'node_01\']\\n    beta = np.asarray(stage[\'beta\'], dtype=float)\\n    fitted = np.asarray(stage[\'fitted\'], dtype=float)\\n    design = np.asarray(stage[\'new_design\'], dtype=float)\\n    _, _, modes = np.linalg.svd(fitted, full_matrices=False)\\n    basis = modes[:2].T\\n    return {\'state_pred\': (design @ (beta @ basis)) @ basis.T}\\n", "node_03": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    y_hat = inputs[\'node_01\'][\'y_hat\']\\n    state_pred = inputs[\'node_02\'][\'state_pred\']\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
