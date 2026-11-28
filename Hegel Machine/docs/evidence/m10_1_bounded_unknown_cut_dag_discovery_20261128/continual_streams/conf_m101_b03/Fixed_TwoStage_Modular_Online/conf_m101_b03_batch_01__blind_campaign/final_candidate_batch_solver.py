"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "0f8c74df2866085a90d9b89c2931fb69fd111b9cd9c74677870e6846ce2fe008"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Invert the reference calibration and predict the response and relaxation modes.\\"\\"\\"\\nfrom itertools import combinations, combinations_with_replacement\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import least_squares, minimize\\n\\n\\ndef _calibrate(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    nxt = np.roll(u, -1, axis=1)\\n    basis = [np.column_stack((np.ones(len(u)), x, z, x*z, x**3))\\n             for x, z in zip(u.T, nxt.T)]\\n    ids = np.arange(len(u))\\n    for _ in range(8):\\n        coef = np.column_stack([np.linalg.lstsq(f[ids], v[ids, j], rcond=None)[0]\\n                                for j, f in enumerate(basis)])\\n        residual = np.column_stack([f @ coef[:, j] for j, f in enumerate(basis)]) - v\\n        ids = np.argsort(np.linalg.norm(residual, axis=1))[:len(u)-3]\\n    return coef\\n\\n\\ndef _inverse(x, c):\\n    def residual(z, row):\\n        w = np.roll(z, -1)\\n        return c[0] + c[1]*z + c[2]*w + c[3]*z*w + c[4]*z**3 - row\\n    return np.array([least_squares(residual, (row-c[0])/c[1], args=(row,),\\n                                   gtol=1e-11).x for row in x])\\n\\n\\ndef _modes(t, traj):\\n    def basis(r):\\n        return -np.expm1(-t[:, None]*r)\\n    def project(h):\\n        return np.linalg.lstsq(h, traj.T, rcond=None)[0]\\n    def residual(r):\\n        h = basis(r)\\n        return (h @ project(h) - traj.T).ravel()\\n    r = least_squares(residual, [.5, 2.], bounds=(.01, 10.)).x\\n    h = basis(np.sort(r))\\n    return h, project(h).T\\n\\n\\ndef _groups(z):\\n    gr=[z]; nonlinear=[]\\n    for a in z.T:\\n        g=[a*a,a**3,a**4]\\n        for k in (.5,1.,1.5,2.,2.5,3.,3.5,4.):\\n            g.extend((np.sin(k*a),np.cos(k*a),np.tanh(k*a),1/(1+k*a*a),a/(1+k*a*a),np.sqrt(1+k*a*a)))\\n        nonlinear.append(np.column_stack(g))\\n    gr.extend(nonlinear)\\n    for i,j in combinations(range(4),2):\\n        a,b=z[:,i],z[:,j]; w=a*b\\n        gr.append(np.column_stack([w]+[w*h for h in (a,b,a*a,b*b,w)]+[f(k*w) for k in (.7,1.5,2.5,3.5) for f in (np.sin,np.cos,np.tanh)]))\\n        gr.append(nonlinear[i]*b[:,None]);gr.append(nonlinear[j]*a[:,None])\\n    gr.append(np.column_stack([np.prod(z[:,p],axis=1) for n in (3,4) for p in combinations_with_replacement(range(4),n) if len(set(p))>=3]))\\n    return gr\\n\\ndef _covariances(gs,a):\\n    k=[]\\n    for g in gs:\\n        f=(g-g[a].mean(0))/np.maximum(g[a].std(0),1e-12)\\n        k.append(f@f.T/f.shape[1])\\n    return np.array(k)\\n\\ndef _predict(k,a,b,y,pen):\\n    ya=y[a]; ya=(ya-ya.mean())/ya.std(); ks=k[:,a][:,:,a]; ks=np.concatenate((ks,np.eye(len(a))[None])); flat=np.ascontiguousarray(ks.reshape(len(ks),-1)); prefer=np.r_[0,np.full(len(ks)-2,-3.),-8.]\\n    def objective(h):\\n        v=np.exp(h); c=cho_factor((v@flat).reshape(len(a),len(a))); w=cho_solve(c,ya); ci=cho_solve(c,np.eye(len(a))); diff=h-prefer\\n        return .5*ya@w+np.log(np.diag(c[0])).sum()+.5*pen*(diff@diff),.5*v*(flat@(ci-np.outer(w,w)).ravel())+pen*diff\\n    fit=minimize(objective,prefer,jac=True,method=\'L-BFGS-B\',bounds=[(-13,5)]*(len(ks)-1)+[(-13,-3)],options={\'maxiter\':120}); w=np.exp(fit.x); c=cho_factor((w@flat).reshape(len(a),len(a)))\\n    return y[a].mean()+y[a].std()*np.einsum(\'k,kij,j->i\',w[:-1],k[:,b][:,:,a],cho_solve(c,ya))\\n\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train, test = study[\'train\'], study[\'test\']\\n    z = _inverse(np.vstack((train[\'x\'], test[\'x\'])), _calibrate(study[\'panel_a\']))\\n    h, modes = _modes(study[\'eval_times\'], train[\'traj\'])\\n    a = np.arange(len(train[\'x\']))\\n    b = np.arange(len(a), len(z))\\n    kernels = _covariances(_groups(z), a)\\n    targets = np.column_stack((train[\'y\'], modes))\\n    prediction = np.column_stack([_predict(kernels, a, b, v, .15) for v in targets.T])\\n    y = prediction[:, 0]\\n    state = prediction[:, 1:] @ h.T\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n"}'

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
    outputs = {'__input__': deepcopy(study)}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        if len(_SPEC_OBJ['topo_order']) == 1 and srcs == ['__input__']:
            node_in = deepcopy(outputs['__input__'])
        else:
            node_in = {s: deepcopy(outputs[s]) for s in srcs}
        in_dig = _canon_digest(node_in)
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Anti-identity-shell check: intermediate node cannot merely echo its raw predecessor dict
            echoed = (out_dig == in_dig) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            if echoed:
                raise RuntimeError(f'DAG node {nid} failed anti-identity-shell check')
            non_identity_checks[nid] = True
        outputs[nid] = out_v
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
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
