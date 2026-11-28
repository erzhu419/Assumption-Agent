"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "874071cd2e88863b4d115ef3ef0c337a6af012a7cccf89f0631b26f7c7982c54", "node_01": "7fdeeaf066c605fd7b3e859029a5cfff4372253328c2eff872e6440522da4694"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Nonlinear panel calibration and additive kernel features for frozen ridge heads.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _coordinates(s, x):\\n    a = s[\'panel_a\']\\n    u, v = a[\'u\'], a[\'v\']\\n    i, j = np.triu_indices(len(u), 1)\\n    off, gain = [], []\\n    for p, q in zip(u.T, v.T):\\n        delta = p[j] - p[i]\\n        ok = np.abs(delta) > 1e-10\\n        slope = np.median((q[j][ok] - q[i][ok])/delta[ok]) if np.any(ok) else 1.\\n        coef = np.array([np.median(q - slope*p), slope])\\n        d = _affine(p[:,None])\\n        for _ in range(4):\\n            r = np.abs(q-d@coef)\\n            keep = r <= max(1e-8, 6*np.median(r))\\n            if keep.sum() < 2:\\n                break\\n            coef = np.linalg.lstsq(d[keep], q[keep], rcond=None)[0]\\n        off.append(coef[0]); gain.append(coef[1] if abs(coef[1]) > 1e-6 else 1.)\\n    cx = (x-off)/gain\\n    # Panel C v1 supports an arctangent near scale 2. Keep the\\n    # saturating channels separate rather than remixing them into v1.\\n    return np.column_stack((cx, np.arctan(2.0*cx)))\\n\\n\\ndef _dist(x):\\n    return (x[:,None,:]-x[None,:,:])**2\\n\\n\\ndef _kernel(sq, t):\\n    d = sq.shape[-1]\\n    exponent = -.5 * sq * np.exp(t[:d])\\n    joint = np.exp(exponent.sum(axis=-1))\\n    add = np.exp(exponent + t[d])/d\\n    return joint, add, exponent\\n\\n\\ndef _learn(sq, y):\\n    n, _, d = sq.shape\\n    sd = np.std(y)\\n    y = (y - np.mean(y))/max(sd,1e-10)\\n    def objective(t):\\n        joint, add, exp = _kernel(sq,t)\\n        k = joint + add.sum(axis=-1)\\n        noise = np.exp(t[-1]); k.flat[::n+1] += noise\\n        f = cho_factor(k, check_finite=False)\\n        alpha = cho_solve(f,y,check_finite=False)\\n        g = .5*(cho_solve(f,np.eye(n),check_finite=False)-np.outer(alpha,alpha))\\n        grad = np.einsum(\'ij,ijk->k\',g,exp*(joint[:,:,None]+add))\\n        return .5*y@alpha+np.log(np.diag(f[0])).sum(), np.r_[grad,np.sum(g*add.sum(axis=-1)),noise*np.trace(g)]\\n    return minimize(objective,np.r_[np.zeros(d),-2.,-9.],jac=True,method=\'L-BFGS-B\',bounds=[(-11.,7.)]*d+[(-10.,5.),(-15.,-2.)],options={\'maxiter\':110}).x\\n\\n\\ndef _gram(sq,t):\\n    j,a,_ = _kernel(sq,t)\\n    return (j+a.sum(axis=-1))/np.exp(t[-1])\\n\\n\\ndef _features(g,n,penalty):\\n    # A training-landmark square root gives exactly the requested kernel ridge fit.\\n    ev, u = np.linalg.eigh(g[:n,:n])\\n    good = ev > max(ev[-1]*1e-12,1e-12)\\n    return g[:,:n] @ (u[:,good]*np.sqrt(penalty/ev[good]))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s = inputs[\'__input__\']; tr = s[\'train\']; n = len(tr[\'x\'])\\n    x = np.concatenate((tr[\'x\'],s[\'test\'][\'x\']))\\n    q = _coordinates(s,x)\\n    sq = _dist(q)\\n    a = tr[\'traj\'] @ np.linalg.svd(tr[\'traj\'],full_matrices=False)[2][0]\\n    ty = _learn(sq[:n,:n],tr[\'y\'])\\n    ta = _learn(sq[:n,:n],a)\\n    gy = _gram(sq,ty)\\n    ga = _gram(sq,ta)\\n    return {\'raw\': _features(0.25*gy + 0.75*ga,n,.014),\\n            \'aligned\': _features(ga,n,.014),\\n            \'hybrid\': _features(gy,n,.007)}\\n", "node_01": "\\"\\"\\"Scalar kernel transfer and separate trajectory-mode regressions.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize\\n\\n\\ndef _ridge(f, y, lam):\\n    k = f @ f[:len(y)].T\\n    return k[len(y):] @ cho_solve(cho_factor(k[:len(y)] + lam*np.eye(len(y))), y)\\n\\n\\ndef _gp(x, n, y):\\n    y0 = y.mean()\\n    scale = max(y.std(), 1e-12)\\n    u = (y-y0)/scale\\n    dist = (x[:, None, :] - x[None, :, :])**2\\n    d = x.shape[1]\\n    def terms(ds, p):\\n        e = -ds*np.exp(p[:d])\\n        joint = np.exp(e.sum(axis=2))\\n        add = np.exp(e+p[d])\\n        return e, joint, add\\n    def loss(p):\\n        e, j, a = terms(dist[:n,:n], p)\\n        noise = np.exp(p[-1])\\n        fact = cho_factor(j+a.sum(axis=2)+noise*np.eye(n))\\n        v = cho_solve(fact, u)\\n        g = .5*(cho_solve(fact,np.eye(n))-np.outer(v,v))\\n        grad = np.einsum(\'ij,ijk->k\',g,e*(j[:,:,None]+a))\\n        return .5*u@v+np.log(np.diag(fact[0])).sum(), np.r_[grad,np.sum(g*a.sum(2)),noise*np.trace(g)]\\n    p = minimize(loss,np.r_[np.zeros(d),-2.,-9.],jac=True,method=\'L-BFGS-B\',bounds=[(-11,7)]*d+[(-10,5),(-15,-2)]).x\\n    _, j, a = terms(dist[:,:n], p)\\n    k = j+a.sum(axis=2)\\n    return y0+scale*k[n:]@cho_solve(cho_factor(k[:n]+np.exp(p[-1])*np.eye(n)),u)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study, parent = inputs[\'__input__\'], inputs[\'node_00\']\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    y = .8*_ridge(parent[\'hybrid\'],train[\'y\'],.007)+.2*_ridge(parent[\'raw\'],train[\'y\'],.012)\\n    x = np.concatenate((train[\'x\'],study[\'test\'][\'x\']))\\n    z = train[\'traj\']\\n    basis = np.linalg.svd(z,full_matrices=False)[2][:2]\\n    state = np.column_stack([_gp(x,n,a) for a in (z@basis.T).T])@basis\\n    return {\'y_hat\':y,\'state_pred\':state,\'diagnostics_vec\':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}\\n"}'

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
