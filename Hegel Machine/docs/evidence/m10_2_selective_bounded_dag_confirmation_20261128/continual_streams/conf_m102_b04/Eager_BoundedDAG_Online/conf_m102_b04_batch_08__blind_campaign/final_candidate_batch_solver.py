"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "7d656eff15f96129cd81360caa27fee67a2e5c036210946537b8f5930332237a", "node_01": "92216b8b99ca0ad49547693878a77185514a6e359370a1481c3482030008cfb2"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\nimport numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _basis(x):\\n    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, j], axis=1) for d in (1, 2, 3) for j in combinations_with_replacement(range(x.shape[1]), d)])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.vstack((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    a, c = study[\'panel_a\'], study[\'panel_c\']\\n    u, v = a[\'u\'], a[\'v\']\\n    f = _affine(u)\\n    fit = np.linalg.lstsq(f, v, rcond=None)[0]\\n    for _ in range(10):\\n        err = np.linalg.norm(f @ fit - v, axis=1)\\n        keep = np.argsort(err)[:int(.82 * len(u))]\\n        fit = np.linalg.lstsq(f[keep], v[keep], rcond=None)[0]\\n    maps = np.linalg.lstsq(_affine(c[\'u\']), c[\'v1\'], rcond=None)[0]\\n    raw = _affine(x) @ maps\\n    calibrated = (x - fit[0]) / fit[1:].diagonal()\\n    aligned = _affine(calibrated) @ maps\\n    squash = lambda q: q / np.sqrt(1 + .09 * q*q)\\n    raw, aligned = _basis(squash(raw)), _basis(squash(aligned))\\n    hybrid = np.column_stack((aligned, _basis(calibrated)))\\n    return {\'raw\': raw, \'aligned\': aligned, \'hybrid\': hybrid}\\n", "node_01": "import numpy as np\\nfrom itertools import combinations_with_replacement\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _panel_fit(panel):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v\'])\\n    n, d = u.shape\\n    a = np.column_stack((np.ones(n), u))\\n    design = [np.column_stack((a, u[:, j] * u[:, (j+1)%d], u[:, j]**3)) for j in range(d)]\\n    keep = np.arange(n)\\n    for _ in range(12):\\n        b = np.column_stack([np.linalg.lstsq(f[keep], v[keep, j], rcond=None)[0] for j, f in enumerate(design)])\\n        err = np.linalg.norm(np.column_stack([f @ b[:, j] for j, f in enumerate(design)]) - v, axis=1)\\n        order = np.argsort(err)\\n        med = np.median(err)\\n        clean = np.flatnonzero(err <= max(6*med, 1e-8))\\n        next_keep = clean if len(clean) >= max(8, int(.8*n)) else np.sort(order[:max(8, int(.85*n))])\\n        if np.array_equal(keep, next_keep):\\n            break\\n        keep = next_keep\\n    return b\\n\\n\\ndef _inverse(raw, b):\\n    n, d = raw.shape\\n    linear = b[1:d+1]\\n    nxt = np.roll(np.arange(d), -1)\\n    col = np.arange(d)\\n    u = (raw-b[0]) @ np.linalg.pinv(linear)\\n    def residual(p):\\n        return b[0]+p @ linear+b[-2]*p*p[:, nxt]+b[-1]*p**3-raw\\n    for _ in range(25):\\n        r = residual(u)\\n        if np.max(np.abs(r), initial=0.) < 1e-11:\\n            break\\n        jac = np.broadcast_to(linear.T, (n,d,d)).copy()\\n        jac[:, col, col] += b[-2]*u[:, nxt]+3*b[-1]*u*u\\n        jac[:, col, nxt] += b[-2]*u\\n        step = (np.linalg.pinv(jac) @ r[..., None])[..., 0]\\n        best = u.copy()\\n        err = np.linalg.norm(r,axis=1)\\n        for scale in (1., .5, .25, .125, .0625):\\n            trial = u-scale*step\\n            trial_err = np.linalg.norm(residual(trial),axis=1)\\n            take = trial_err < err\\n            best[take], err[take] = trial[take], trial_err[take]\\n        u = best\\n    return u, np.max(np.abs(residual(u)), axis=1)\\n\\n\\ndef _predict(u, y, z):\\n    n = len(y)\\n    one = np.ones(len(u))\\n    square = np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(u.shape[1]),2)])\\n    base_y = np.column_stack((one,u,square))\\n    base_z = np.column_stack((one,u))\\n    cube = u**3\\n    def profile(target, base, power, bound):\\n        def fit(a):\\n            f = np.column_stack((base,cube/(1+a*np.abs(u)**power)))\\n            b = np.linalg.lstsq(f[:n], target, rcond=None)[0]\\n            return f @ b\\n        def loss(a):\\n            return np.mean((fit(a)[:n]-target)**2)\\n        a = minimize_scalar(loss, method=\'bounded\', bounds=(.002,bound)).x\\n        return fit(a)[n:]\\n    return profile(y,base_y,3,2.), profile(z,base_z,2,3.)\\n\\n\\ndef _ridge_fallback(f, targets):\\n    n = len(targets)\\n    a = f[:n]\\n    return f[n:] @ np.linalg.solve(a.T @ a + .018*np.eye(a.shape[1]),a.T @ targets)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    frozen = inputs[\'node_00\']\\n    hybrid = np.asarray(frozen[\'hybrid\'])\\n    aligned = np.asarray(frozen[\'aligned\'])\\n    train, test = study[\'train\'], study[\'test\']\\n    y, z = np.asarray(train[\'y\']), np.asarray(train[\'traj\'])\\n    raw = np.vstack((train[\'x\'], test[\'x\']))\\n    u, err = _inverse(raw, _panel_fit(study[\'panel_a\']))\\n    if hybrid.shape[0] != len(raw) or aligned.shape[0] != len(raw):\\n        raise ValueError(\'Frozen feature rows do not match the study\')\\n    bad = (~np.isfinite(u).all(axis=1)) | (err > 1e-4)\\n    if np.any(bad[:len(y)]):\\n        yh, state = _ridge_fallback(hybrid,y), _ridge_fallback(aligned,z)\\n    else:\\n        yh, state = _predict(np.where(np.isfinite(u),u,0.), y,z)\\n        fail = bad[len(y):]\\n        if np.any(fail):\\n            yh[fail] = _ridge_fallback(hybrid,y)[fail]\\n            state[fail] = _ridge_fallback(aligned,z)[fail]\\n    end = state[:, -1]\\n    return {\'y_hat\':yh, \'state_pred\':state, \'diagnostics_vec\':np.array([yh.mean(),yh.std(),end.mean(),end.std()])}\\n"}'

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
