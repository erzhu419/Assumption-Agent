"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "7d656eff15f96129cd81360caa27fee67a2e5c036210946537b8f5930332237a", "node_01": "3e3a97687d203c6a025b6f6938455a8f9fb736c86ea886ef8b51d309afa539cc"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\nimport numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _basis(x):\\n    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, j], axis=1) for d in (1, 2, 3) for j in combinations_with_replacement(range(x.shape[1]), d)])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.vstack((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    a, c = study[\'panel_a\'], study[\'panel_c\']\\n    u, v = a[\'u\'], a[\'v\']\\n    f = _affine(u)\\n    fit = np.linalg.lstsq(f, v, rcond=None)[0]\\n    for _ in range(10):\\n        err = np.linalg.norm(f @ fit - v, axis=1)\\n        keep = np.argsort(err)[:int(.82 * len(u))]\\n        fit = np.linalg.lstsq(f[keep], v[keep], rcond=None)[0]\\n    maps = np.linalg.lstsq(_affine(c[\'u\']), c[\'v1\'], rcond=None)[0]\\n    raw = _affine(x) @ maps\\n    calibrated = (x - fit[0]) / fit[1:].diagonal()\\n    aligned = _affine(calibrated) @ maps\\n    squash = lambda q: q / np.sqrt(1 + .09 * q*q)\\n    raw, aligned = _basis(squash(raw)), _basis(squash(aligned))\\n    hybrid = np.column_stack((aligned, _basis(calibrated)))\\n    return {\'raw\': raw, \'aligned\': aligned, \'hybrid\': hybrid}\\n", "node_01": "\\"\\"\\"Invert the panel calibration, then fit separate scalar and trajectory heads.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\nimport numpy as np\\n\\n\\ndef _calibration(u, v, j):\\n    # The calibration couples each coordinate to its cyclic successor.\\n    nxt = np.roll(u, -1, axis=1)\\n    f = np.stack((np.ones_like(u), u, nxt, u * nxt, u**3), axis=-1)\\n    def fit(keep):\\n        return np.column_stack([np.linalg.lstsq(f[keep, k], v[keep, k], rcond=None)[0] for k in range(u.shape[1])])\\n    def residual(a):\\n        return np.linalg.norm(np.einsum(\'nkp,pk->nk\', f, a) - v, axis=1)\\n    n = len(u)\\n    m = max(5, int(.82 * n))\\n    best = None\\n    for keep in (np.arange(n), j[:m]):\\n        for _ in range(10):\\n            a = fit(keep)\\n            new = np.sort(np.argsort(residual(a))[:m])\\n            if np.array_equal(new, keep):\\n                break\\n            keep = new\\n        e = residual(a)\\n        score = np.mean(np.sort(e)[:m]**2)\\n        if best is None or score < best[0]:\\n            best = score, a\\n    a = best[1]\\n    return a if np.median(residual(a)) <= .015 * max(np.std(v), 1e-8) else None\\n\\n\\ndef _inverse(x, q, u, v):\\n    f = np.column_stack((np.ones(len(u)), u))\\n    affine = np.linalg.lstsq(f, v, rcond=None)[0]\\n    order = np.argsort(np.linalg.norm(f @ affine - v, axis=1))\\n    a = _calibration(u, v, order)\\n    if a is None:\\n        return None\\n    d = x.shape[1]\\n    k = np.arange(d)\\n    nxt = (k + 1) % d\\n    def error(w):\\n        z = w[:, nxt]\\n        return a[0] + a[1]*w + a[2]*z + a[3]*w*z + a[4]*w**3 - x\\n    jac = np.zeros((len(x), d, d))\\n    try:\\n        jac[:, k, k] = a[1]\\n        jac[:, k, nxt] = a[2]\\n        w = np.linalg.solve(jac[0], (x-a[0]).T).T\\n        use_q = np.linalg.norm(error(q), axis=1) < np.linalg.norm(error(w), axis=1)\\n        w[use_q] = q[use_q]\\n        for _ in range(24):\\n            r = error(w)\\n            if np.max(np.abs(r)) <= 1e-10:\\n                return w\\n            jac[:, k, k] = a[1] + a[3]*w[:, nxt] + 3*a[4]*w*w\\n            jac[:, k, nxt] = a[2] + a[3]*w\\n            step = np.linalg.solve(jac, r[..., None])[..., 0]\\n            norm = np.linalg.norm(r, axis=1)\\n            for j in range(12):\\n                trial = w - step * (0.5**j)\\n                good = np.linalg.norm(error(trial), axis=1) < norm\\n                if j == 0:\\n                    update = w.copy()\\n                    pending = np.ones(len(w), dtype=bool)\\n                good &= pending\\n                update[good] = trial[good]\\n                pending[good] = False\\n                if not pending.any():\\n                    break\\n            w = update\\n        if np.max(np.abs(error(w))) <= 1e-7:\\n            return w\\n    except np.linalg.LinAlgError:\\n        pass\\n    return None\\n\\n\\ndef _basis(w, degree, fallback=False):\\n    cols = [np.ones(len(w))]\\n    for d in range(1, degree+1):\\n        for j in combinations_with_replacement(range(w.shape[1]), d):\\n            weight = 0.5**(len(set(j))-1 + 0.5*max(d-2, 0)) if fallback else 1.0\\n            cols.append(weight * np.prod(w[:, j], axis=1))\\n    return np.column_stack(cols)\\n\\n\\ndef _ridge(f, y, penalty):\\n    n = len(y)\\n    a, b = f[:n], f[n:]\\n    if a.shape[1] <= n:\\n        return b @ np.linalg.solve(a.T @ a + penalty*np.eye(a.shape[1]), a.T @ y)\\n    return (b @ a.T) @ np.linalg.solve(a @ a.T + penalty*np.eye(n), y)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    upstream = inputs[\'node_00\']\\n    train, test = study[\'train\'], study[\'test\']\\n    x = np.vstack((train[\'x\'], test[\'x\']))\\n    start = upstream[\'aligned\'].shape[1] + 1\\n    q = upstream[\'hybrid\'][:, start:start+x.shape[1]]\\n    panel = study[\'panel_a\']\\n    w = _inverse(x, q, panel[\'u\'], panel[\'v\'])\\n    traj = train[\'traj\']\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    if w is None:\\n        y = _ridge(_basis(q / np.sqrt(1 + .60*q*q), 6, True), train[\'y\'], .0003)\\n        state = _ridge(_basis(q / np.sqrt(1 + .85*q*q), 6, True), traj @ modes.T, .00146) @ modes\\n    else:\\n        y = _ridge(_basis(w, 2), train[\'y\'], .0015)\\n        state = _ridge(_basis(w, 1), traj @ modes.T, .0015) @ modes\\n    return {\'y_hat\': y, \'state_pred\': state, \'diagnostics_vec\': np.array([\\n        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}\\n"}'

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
