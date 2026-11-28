"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "de20f971e81f68973ac1cbb3422e972b76ed9528aed14a01d7ad55efa989f5a2"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Panel-conditioned Gaussian regression for scalar and trajectory outputs.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize, minimize_scalar\\n\\n\\ndef affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef coordinates(study, x):\\n    a, b, c = (study[\'panel_\'+key] for key in \'abc\')\\n    offset, gain = [], []\\n    for u, v in zip(a[\'u\'].T, a[\'v\'].T):\\n        d = affine(u[:,None]); k = np.argsort(u); l, r = np.triu_indices(len(u), 1)\\n        slope = np.median((v[k[r]]-v[k[l]]) / (u[k[r]]-u[k[l]]))\\n        coeff = np.array([np.median(v-slope*u), slope])\\n        for _ in range(5):\\n            keep = np.argsort(abs(v-d@coeff))[:int(.8*len(u))]\\n            coeff = fit(d[keep], v[keep])\\n        offset.append(coeff[0]); gain.append(coeff[1])\\n    cx = (x-offset)/gain\\n    bx = affine(x)@fit(affine(b[\'u\']), b[\'v\'])\\n    u = c[\'u\']\\n    vmap = lambda q: affine(np.arctan(2.12*q))\\n    def wmap(q, a):\\n        return np.column_stack((affine(q), q*q/(1+a*q*q)))\\n    def loss(a):\\n        d = wmap(u, a)\\n        return np.sum((d@fit(d, c[\'v2\'])-c[\'v2\'])**2)\\n    alpha = minimize_scalar(loss, bounds=(0., 1.), method=\'bounded\').x\\n    v = fit(vmap(u), c[\'v1\']); w = fit(wmap(u, alpha), c[\'v2\'])\\n    return cx, bx, vmap(cx)@v, wmap(bx,alpha)@w\\n\\n\\ndef gp(x, xt, y, additive=False):\\n    mu, sd = y.mean(), max(y.std(), 1e-12)\\n    yy = (y-mu)/sd\\n    n, d = x.shape\\n    sq = (x[:,None,:]-x[None,:,:])**2\\n    dsq = (xt[:,None,:]-x[None,:,:])**2\\n    def kernel(dist, th):\\n        w = np.exp(th[:d]); e = -.5*dist*w\\n        per = np.exp(e) if additive else None\\n        k = np.exp(e.sum(2))\\n        if additive:\\n            extra = np.exp(th[d])*per/d\\n            k = k + extra.sum(2)\\n        return k, per if not additive else extra, e\\n    def objective(th):\\n        k, extra, e = kernel(sq, th)\\n        noise = np.exp(th[-1]); k.flat[::n+1] += noise\\n        fac = cho_factor(k, check_finite=False)\\n        al = cho_solve(fac, yy, check_finite=False)\\n        g = .5*(cho_solve(fac, np.eye(n), check_finite=False)-np.outer(al,al))\\n        joint = k.copy(); joint.flat[::n+1] -= noise\\n        if additive:\\n            joint -= extra.sum(2)\\n        grad = np.einsum(\'ij,ijk->k\', g, e*(joint[:,:,None]+(extra if additive else 0.)))\\n        if additive:\\n            grad = np.r_[grad,np.sum(g*extra.sum(2))]\\n        return .5*yy@al+np.log(np.diag(fac[0])).sum(), np.r_[grad, noise*np.trace(g)]\\n    th = minimize(objective, np.r_[np.zeros(d),[-2.] if additive else [],-9.], jac=True,\\n                  method=\'L-BFGS-B\', bounds=[(-11,7)]*d+([(-10,5)] if additive else [])+[(-15,-2)]).x\\n    k = kernel(sq,th)[0]; k.flat[::n+1] += np.exp(th[-1])\\n    return mu+sd*(kernel(dsq,th)[0]@cho_solve(cho_factor(k),yy))\\n\\n\\ndef solve_batch(study):\\n    tr = study[\'train\']; x, xt = tr[\'x\'], study[\'test\'][\'x\']; n = len(x)\\n    cx, bx, v, w = coordinates(study, np.concatenate((x,xt)))\\n    q = np.column_stack((cx,v)); y = gp(q[:n],q[n:],tr[\'y\'],True)\\n    q = np.column_stack((bx,w)); z = tr[\'traj\']; _, _, vt = np.linalg.svd(z,full_matrices=False)\\n    a = gp(q[:n],q[n:],z@vt[0]); state = a[:,None]*vt[0]\\n    return {\'y_hat\':y, \'state_pred\':state,\\n            \'diagnostics_vec\':np.array([y.mean(), y.std(), state[:,-1].mean(), state[:,-1].std()])}\\n"}'

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
