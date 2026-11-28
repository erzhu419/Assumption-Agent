"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_14", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "5554671d1fcb9c270ded89023edcb920fa0c029e2c491ea349cf9557df0b6c5e", "node_01": "8dcdc3f5e73aa7d75f6cda4b3db2105a6350a813731e0a782a62463c79468b7a"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _profile(loss, lo, hi):\\n    grid = np.linspace(lo, hi, 49)\\n    i = int(np.argmin([loss(g) for g in grid]))\\n    return minimize_scalar(loss, method=\'bounded\',\\n                           bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),\\n                           options={\'xatol\': 1e-9}).x\\n\\n\\ndef _invert_a(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    d = _aff(u)\\n    fit = _ls(d, v)\\n    for _ in range(5):\\n        err = np.linalg.norm(d @ fit - v, axis=1)\\n        keep = np.argsort(err)[:max(6, int(.875*len(u)))]\\n        fit = _ls(d[keep], v[keep])\\n    err = np.linalg.norm(d @ fit - v, axis=1)\\n    keep = err <= max(4*np.median(err), .02)\\n    u, v = u[keep], v[keep]\\n    n = u.shape[1]\\n    j = np.arange(n)\\n    k = (j+2) % n\\n\\n    def basis(z, g):\\n        sn = np.sin(g*z)\\n        return np.stack((np.ones_like(z), z, z[:, k], sn, sn[:, k], z**3), axis=-1)\\n\\n    def fit_at(g, coeff=False):\\n        a = basis(u, g)\\n        q = np.array([_ls(a[:, i], v[:, i]) for i in j])\\n        return q if coeff else np.mean((np.sum(a*q, axis=-1)-v)**2)\\n\\n    g = _profile(fit_at, .8, 4.5)\\n    q = fit_at(g, True)\\n    z = (x-q[:, 0])/q[:, 1]\\n    jac = np.zeros((len(x), n, n))\\n    for _ in range(30):\\n        r = np.sum(basis(z, g)*q, axis=-1)-x\\n        err = np.sum(r*r, axis=1)\\n        if np.max(err, initial=0.) < 1e-24:\\n            break\\n        cs = g*np.cos(g*z)\\n        jac[:, j, j] = q[:, 1]+q[:, 3]*cs+3*q[:, 5]*z*z\\n        jac[:, j, k] = q[:, 2]+q[:, 4]*cs[:, k]\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        step /= np.maximum(1, np.max(abs(step), axis=1, keepdims=True)/.5)\\n        pending = err > 1e-24\\n        for _ in range(15):\\n            trial = z-step\\n            rr = np.sum(basis(trial, g)*q, axis=-1)-x\\n            take = pending & (np.sum(rr*rr, axis=1) < err)\\n            z[take] = trial[take]\\n            pending &= ~take\\n            if not np.any(pending):\\n                break\\n            step *= .5\\n    return z\\n\\n\\ndef _map_b(panel, z):\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def basis(x, g):\\n        return _aff(np.cos(g*x))\\n\\n    def loss(g):\\n        d = basis(u, g)\\n        return np.mean((d @ _ls(d, v)-v)**2)\\n\\n    g = _profile(loss, .7, 4.5)\\n    return basis(z, g) @ _ls(basis(u, g), v)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']), axis=0)\\n    z = _invert_a(study[\'panel_a\'], x)\\n    w = _map_b(study[\'panel_b\'], z)\\n    # A shared monotone coordinate stretch conditions the frozen head\'s\\n    # saturating features without changing panel-B symmetry or mixing channels.\\n    return {\'w\': w + 0.27 * w**3}\\n", "node_01": "import itertools\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _features(w, gain, interaction, kind):\\n    u = gain * w\\n    if kind == \'atan\':\\n        z = np.arctan(u)\\n    elif kind == \'inv3\':\\n        z = np.sinh(np.arcsinh(u) / 3)\\n    else:\\n        z = np.arcsinh(u)\\n    if not interaction:\\n        return np.column_stack((_aff(w), z))\\n    pairs = [z[:, i] * z[:, j] for i, j in itertools.combinations_with_replacement(range(w.shape[1]), 2)]\\n    return np.column_stack((_aff(w), z, *pairs))\\n\\n\\ndef _predict(w, target, interaction, kind):\\n    n = len(target)\\n    scale = np.maximum(np.std(target, axis=0), 1e-8)\\n    def loss(gain):\\n        f = _features(w[:n], gain, interaction, kind)\\n        r = (f @ _ls(f, target) - target) / scale\\n        return np.mean(r*r)\\n    gain = minimize_scalar(loss, bounds=(0.8, 7), method=\'bounded\').x\\n    f = _features(w, gain, interaction, kind)\\n    return f[n:] @ _ls(f[:n], target)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    tr = inputs[\'__input__\'][\'train\']\\n    w = inputs[\'node_00\'][\'w\']\\n    y = (0.35 * _predict(w, tr[\'y\'], True, \'asinh\')\\n         + 0.65 * _predict(w, tr[\'y\'], True, \'atan\'))\\n    p = _predict(w, tr[\'traj\'], False, \'inv3\')\\n    tail = p[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': p, \'diagnostics_vec\': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}\\n"}'

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
