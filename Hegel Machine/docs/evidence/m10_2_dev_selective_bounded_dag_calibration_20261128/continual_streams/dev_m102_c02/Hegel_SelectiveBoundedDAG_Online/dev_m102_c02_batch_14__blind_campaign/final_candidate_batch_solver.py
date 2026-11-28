"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_15", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "412847fcdb7fde5079bc37343a8b2555e800f2cc52b2c0bd37c920ac81d42dc5", "node_01": "a384036848a9cb2bfc6de7b8c9b084f5e41c888a0f22db426a41dd2d180e82dd"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates from panel A and apply the panel-B operator.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar, root, minimize\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _latent(panel, x):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    opposite = (np.arange(u.shape[1]) + 2) % u.shape[1]\\n    residual = np.linalg.norm(v - _affine(u) @ _ls(_affine(u), v), axis=1)\\n    keep = residual < max(6 * np.median(residual), 0.1)\\n    if keep.sum() >= 8:\\n        u, v = u[keep], v[keep]\\n\\n    def basis(z, p):\\n        return np.stack((np.ones_like(z), z, np.tanh(p[0]*z), np.sin(p[1]*z[:, opposite])), axis=-1)\\n\\n    def project(p, return_coef=False):\\n        a = basis(u, p)\\n        coef = np.stack([_ls(a[:, j], v[:, j]) for j in range(u.shape[1])])\\n        return coef if return_coef else (np.sum(a*coef, axis=-1)-v).ravel()\\n\\n    p = least_squares(project, [1.8, 2.4], bounds=([0.3, 0.8], [4., 4.])).x\\n    coef = project(p, True)\\n    return np.stack([root(lambda z: np.sum(basis(z[None], p)[0]*coef, axis=-1)-q, q).x for q in x])\\n\\n\\ndef _panel_b(panel, z):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    def basis(q, a):\\n        return np.c_[_affine(q / (1 + a*q*q)), q]\\n    def loss(a):\\n        d = basis(u, a)\\n        return np.mean((d @ _ls(d, v) - v)**2)\\n    a = minimize_scalar(loss, bounds=(.05, 4.), method=\'bounded\').x\\n    return basis(z, a) @ _ls(basis(u, a), v)\\n\\n\\ndef _gram_blocks(study, x, n):\\n    z = _latent(study[\'panel_a\'], x)\\n    b = _panel_b(study[\'panel_b\'], z)\\n    # Separate first-order, mixed second-order and smooth remainder priors.\\n    def gram(f):\\n        f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0),1.e-7)\\n        return (f@f.T)/f.shape[1]\\n    h = float(np.asarray(study[\'telemetry_vec\'])[3])\\n    zc = gram(z)\\n    bc = gram(b)\\n    rc = gram(z/(1+h*z*z))\\n    gc = gram(z**3/(1+h*z*z))\\n    ec = gram(z*z)\\n    sc = gram(np.sin(float(np.asarray(study[\'telemetry_vec\'])[1])*z))\\n    pc = []\\n    p=study[\'panel_c\'];u=np.asarray(p[\'u\'],float)\\n    for key, basis in [(\'v1\',lambda a:np.c_[_affine(a),np.arctan(2.4*a),a**3]),(\'v2\',lambda a:np.c_[_affine(a),a*a,a**3])]:\\n        pc.append(gram(basis(z)@_ls(basis(u),np.asarray(p[key],float))))\\n    fc=(zc+bc)/2\\n    dc=np.maximum(fc.diagonal()[:,None]+fc.diagonal()[None,:]-2*fc,0.)\\n    blocks=[zc,bc,rc,gc,ec,sc,*pc,zc*zc,zc*bc,bc*bc,zc*gc,gc*gc,zc*ec,bc*ec,bc*pc[0],bc*pc[1],fc**3,fc**4,fc**5,np.exp(-.3*dc),np.exp(-.1*dc)]\\n    return np.stack([k/max(np.mean(k.diagonal()[:n]),1.e-10) for k in blocks])\\n\\n\\ndef _smooth_models(grams, response):\\n    from scipy.linalg import cho_factor, cho_solve\\n    from scipy.optimize import minimize\\n    n=len(response); eye=np.eye(n); ones=np.ones((n,1));\\n    avg=response.mean(); sd=max(response.std(),1.e-7); y=(response-avg)/sd\\n    gs=np.ascontiguousarray(grams[:,:n,:n]); flat=gs.reshape(len(gs),-1)\\n    def solve(w, full=False):\\n        k=(w[:-1]@flat).reshape(n,n)+w[-1]*eye+1.\\n        fact=cho_factor(k,check_finite=False); ki=cho_solve(fact,eye,check_finite=False); a=ki@y\\n        loo=y-a/np.diag(ki)\\n        if full:\\n            pred=(np.einsum(\'k,kij->ij\',w[:-1],grams[:,:,:n])+1.)@a\\n            return np.mean(np.abs(y-loo)),pred*sd+avg,loo*sd+avg\\n        v=ki-np.outer(a,a)\\n        val=.5*(y@a)+np.log(np.diag(fact[0])).sum()\\n        grad=np.r_[.5*(flat@v.ravel()),.5*np.trace(v)]*w\\n        return val,grad\\n    def objective(a): return solve(np.exp(a))\\n    results=[]\\n    for initial in (.003,.03):\\n        w0=np.full(len(gs)+1,.04);w0[-1]=initial\\n        opt=minimize(objective,np.log(w0),jac=True,method=\'L-BFGS-B\',bounds=[(-13,4)]*len(gs)+[(-11,-.65)],options={\'maxiter\':145,\'ftol\':2e-8})\\n        results.append(solve(np.exp(opt.x),True))\\n    for ii in ([0,1,8,9,10],[0,1,8,9,10,17],[0,1,3,8,9,10,11,12,17],[0,1,2,3,8,9,10,11,12],[0,1,8,9,10,17,18],[0,1,8,9,10,17,18,19],[0,1,8,9,10,20,21],list(range(len(gs)))):\\n        for order in (.1,1.):\\n            for noise in (2.e-5,.0002,.002,.01,.04):\\n                w=np.full(len(gs)+1,0.);w[ii]=1./len(ii);w[np.array(ii)[np.array(ii)>=17]]*=order;w[-1]=noise\\n                results.append(solve(w,True))\\n    results.sort(key=lambda q:q[0]); best=results[0][0]\\n    sel=[q for q in results[:12] if q[0]<best*1.10+1.e-8]\\n    wt=np.exp(-35*(np.array([q[0] for q in sel])/max(best,1.e-8)-1)); wt/=wt.sum()\\n    return sum(q[1]*w for q,w in zip(sel,wt)),sum(q[2]*w for q,w in zip(sel,wt)),best*sd\\n\\n\\ndef _scalar_profile(grams, y, amplitude, fits):\\n    \\"\\"\\"Fit the scalar-specific residual while profiling the shared modes.\\"\\"\\"\\n    from scipy.linalg import cho_factor, cho_solve\\n    from scipy.optimize import minimize\\n\\n    n = len(y)\\n    ymean, yscale = y.mean(), max(y.std(), 1.e-10)\\n    target = (y-ymean)/yscale\\n    center = amplitude.mean(0)\\n    scale = np.maximum(amplitude.std(0), 1.e-8)\\n    observed = (amplitude-center)/scale\\n    smooth = (np.column_stack([f[0] for f in fits])-center)/scale\\n    loo_amp = (np.column_stack([f[1] for f in fits])-center)/scale\\n    eye = np.eye(n)\\n    candidates = []\\n    base = (grams[0]+grams[1])/2\\n    for nonlinear in (False, True):\\n        def shared(a):\\n            m = _affine(a)\\n            if nonlinear:\\n                u = a[:,0]\\n                m = np.c_[m, u/(1.+.7*u*u), u*a[:,1]]\\n            return m\\n        design = shared(observed)\\n        pred_design = shared(smooth)\\n        loo_design = shared(loo_amp)\\n        delta = loo_design-design\\n        # Broad prior on the linear shared terms; shrink optional nonlinear terms.\\n        prior = np.array([1., 2., 1.] + ([.1, .02] if nonlinear else []))\\n        cov_shared = (design*prior)@design.T\\n        blocks = np.stack([grams[i] for i in (0,1,2,3,4,5,8,9,10,11,13,14,16,17,18)] + [base**3, base**4])\\n        flat = np.ascontiguousarray(blocks[:,:n,:n]).reshape(len(blocks),-1)\\n        mat = np.ascontiguousarray(blocks[:,:,:n])\\n        def solve(w, predict=False):\\n            k = cov_shared+(w[:-1]@flat).reshape(n,n)+w[-1]*eye\\n            factor = cho_factor(k,check_finite=False)\\n            inv = cho_solve(factor,eye,check_finite=False)\\n            coef = inv@target\\n            if predict:\\n                diag = inv.diagonal()\\n                # Sherman-Morrison deletion for the profiled shared coefficients.\\n                beta = prior*(design.T@coef)\\n                deletion = ((inv@design)*prior)*(coef/diag)[:,None]\\n                out = target-coef/diag+delta@beta-np.sum(delta*deletion,axis=1)\\n                full = pred_design@beta+np.einsum(\'k,kij,j->i\',w[:-1],mat,coef)\\n                return np.mean(abs(out-target))*yscale, full*yscale+ymean, out*yscale+ymean\\n            score = .5*(target@coef)+np.log(np.diag(factor[0])).sum()\\n            v = inv-np.outer(coef,coef)\\n            gradient = .5*np.r_[flat@v.ravel(),np.trace(v)]*w\\n            return score,gradient\\n        for start in (.002,.015):\\n            init = np.full(len(blocks)+1,.015); init[-1]=start\\n            opt = minimize(lambda p: solve(np.exp(p)),np.log(init),jac=True,method=\'L-BFGS-B\',bounds=[(-13.,2.)]*len(blocks)+[(-10.,-3.)],options={\'maxiter\':120,\'ftol\':1.e-8})\\n            candidates.append(solve(np.exp(opt.x),True))\\n        for width in (.03,.15,.6):\\n            for nugget in (.0001,.001,.006):\\n                w = np.zeros(len(blocks)+1)\\n                w[[0,1,6,7,8,13,14]] = width*np.array([1,1,1,1,1,.15,.025])\\n                w[-1] = nugget\\n                candidates.append(solve(w,True))\\n    candidates.sort(key=lambda m:m[0]); top=candidates[:8]\\n    error=np.array([m[0] for m in top]); w=np.exp(-19.*(error/error.min()-1));w/=w.sum()\\n    full=sum(v[1]*a for v,a in zip(top,w)); loo=sum(v[2]*a for v,a in zip(top,w))\\n    return full,loo,float(np.mean(abs(loo-y)))\\n\\n\\ndef _coordinates(study, x, q):\\n    n=len(x); g=_gram_blocks(study,np.r_[x,q],n)\\n    traj=np.asarray(study[\'train\'][\'traj\'],float); y=np.asarray(study[\'train\'][\'y\'],float)\\n    _,_,v=np.linalg.svd(traj,full_matrices=False)\\n    amp=traj@v[:2].T\\n    fits=[_smooth_models(g,amp[:,j]) for j in range(2)]\\n    ys=_smooth_models(g,y)\\n    # Estimate the shared component separately from the scalar-specific response.\\n    # Only out-of-fold model estimates are used to pick the decomposition strength.\\n    yt=[]\\n    for coupling in (.5,1.):\\n        aa=_affine(amp)\\n        beta=_ls(aa,y)*coupling\\n        res=_smooth_models(g,y-aa@beta)\\n        full=res[0]+_affine(np.column_stack([f[0] for f in fits]))@beta\\n        loo=res[1]+_affine(np.column_stack([f[1] for f in fits]))@beta\\n        yt.append((full,loo,float(np.mean(abs(loo-y)))))\\n    profile = _scalar_profile(g,y,amp,fits)\\n    yy=[ys,*yt,profile]; err=np.array([k[2] for k in yy]); wt=np.exp(-13*(err/err.min()-1));wt/=wt.sum()\\n    scalar=sum(k[0]*w for k,w in zip(yy,wt))\\n    c=np.column_stack((scalar, fits[0][0], fits[1][0], np.zeros(len(scalar))))\\n    return c, (err, [f[2] for f in fits])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study=inputs[\'__input__\']; x=np.asarray(study[\'train\'][\'x\'],float); q=np.asarray(study[\'test\'][\'x\'],float)\\n    c,_=_coordinates(study,x,q)\\n    return {\'c\':c[:len(x)],\'query\':c[len(x):]}\\n", "node_01": "\\"\\"\\"Scalar and trajectory readouts on the frozen calibrated coordinates.\\"\\"\\"\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _scalar_inputs(c):\\n    paired = np.sum(c[:, :2] * c[:, 2:], axis=1)\\n    return np.column_stack((c, paired))\\n\\n\\ndef _scalar_readout(c, query, y):\\n    u, uq = _scalar_inputs(c), _scalar_inputs(query)\\n    x, xq = _affine(u), _affine(uq)\\n\\n    def design(base, z, w):\\n        s = z @ w\\n        return np.column_stack((base, s / (1.0 + s * s)))\\n\\n    def projected_residual(w):\\n        d = design(x, u, w)\\n        return d @ np.linalg.lstsq(d, y, rcond=None)[0] - y\\n\\n    linear = np.linalg.lstsq(x, y, rcond=None)[0][1:]\\n    if np.linalg.norm(linear) < 1e-10:\\n        return xq @ np.linalg.lstsq(x, y, rcond=None)[0]\\n    direction = linear / np.linalg.norm(linear)\\n    optimum = min((least_squares(projected_residual, scale * direction,\\n                                 bounds=(-5., 5.), max_nfev=160, gtol=2e-10)\\n                   for scale in (0.6, 1.2, 2.0)), key=lambda fit: fit.cost)\\n    d = design(x, u, optimum.x)\\n    beta = np.linalg.lstsq(d, y, rcond=None)[0]\\n    return design(xq, uq, optimum.x) @ beta\\n\\n\\ndef _trajectory(c, query, traj, times, intercept=True):\\n    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)\\n    beta = np.linalg.lstsq(design, traj, rcond=None)[0]\\n    fitted = design @ beta\\n    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)\\n    t = times / scale\\n\\n    def basis(log_rates):\\n        return -np.expm1(-np.exp(log_rates[:, None]) * t)\\n\\n    def residual(log_rates):\\n        modes = basis(log_rates)\\n        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()\\n\\n    log_rates = least_squares(residual, np.log([0.6, 2.7]),\\n                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),\\n                              max_nfev=100).x\\n    modes = basis(log_rates)\\n    return new_design @ (beta @ np.linalg.pinv(modes)) @ modes\\n\\n\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coordinates = inputs[\'node_00\']\\n    c = np.asarray(coordinates[\'c\'], dtype=float)\\n    query = np.asarray(coordinates[\'query\'], dtype=float)\\n    train = study[\'train\']\\n    y_hat = _scalar_readout(c, query, np.asarray(train[\'y\'], dtype=float))\\n    state_pred = _trajectory(c, query, np.asarray(train[\'traj\'], dtype=float),\\n                             np.asarray(study[\'eval_times\'], dtype=float))\\n    final_state = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(),\\n                                         final_state.mean(), final_state.std()])}\\n"}'

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
