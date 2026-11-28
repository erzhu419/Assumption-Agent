"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "7d656eff15f96129cd81360caa27fee67a2e5c036210946537b8f5930332237a", "node_01": "e1178e8b2099edaf0ebf816134ec65372f8375849b42e1aaa6093614f76dcc8c"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\nimport numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _basis(x):\\n    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, j], axis=1) for d in (1, 2, 3) for j in combinations_with_replacement(range(x.shape[1]), d)])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.vstack((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    a, c = study[\'panel_a\'], study[\'panel_c\']\\n    u, v = a[\'u\'], a[\'v\']\\n    f = _affine(u)\\n    fit = np.linalg.lstsq(f, v, rcond=None)[0]\\n    for _ in range(10):\\n        err = np.linalg.norm(f @ fit - v, axis=1)\\n        keep = np.argsort(err)[:int(.82 * len(u))]\\n        fit = np.linalg.lstsq(f[keep], v[keep], rcond=None)[0]\\n    maps = np.linalg.lstsq(_affine(c[\'u\']), c[\'v1\'], rcond=None)[0]\\n    raw = _affine(x) @ maps\\n    calibrated = (x - fit[0]) / fit[1:].diagonal()\\n    aligned = _affine(calibrated) @ maps\\n    squash = lambda q: q / np.sqrt(1 + .09 * q*q)\\n    raw, aligned = _basis(squash(raw)), _basis(squash(aligned))\\n    hybrid = np.column_stack((aligned, _basis(calibrated)))\\n    return {\'raw\': raw, \'aligned\': aligned, \'hybrid\': hybrid}\\n", "node_01": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\n\\n\\ndef _quadratic(x):\\n    return np.column_stack((np.ones(len(x)), x, *(x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2))))\\n\\n\\ndef _calibration(panel):\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n    prev = np.roll(np.arange(u.shape[1]), -1)\\n    features = [np.column_stack((np.ones(len(u)), u[:, j], u[:, prev[j]], u[:, j] * u[:, prev[j]], u[:, j] ** 3)) for j in range(u.shape[1])]\\n    weights = np.ones(len(u))\\n    for _ in range(12):\\n        root = np.sqrt(weights)\\n        coef = np.array([np.linalg.lstsq(a * root[:, None], v[:, j] * root, rcond=None)[0] for j, a in enumerate(features)])\\n        errors = np.linalg.norm(np.column_stack([a @ b for a, b in zip(features, coef)]) - v, axis=1)\\n        scale = max(3.0 * np.median(errors), 1e-4)\\n        weights = 1 / (1 + (errors / scale) ** 2)\\n    keep = errors <= max(8 * np.median(errors), 5e-4)\\n    if keep.sum() >= max(8, len(u) // 2):\\n        coef = np.array([np.linalg.lstsq(a[keep], v[keep, j], rcond=None)[0] for j, a in enumerate(features)])\\n    return coef, prev\\n\\n\\ndef _invert(x, warm, coef, prev):\\n    bias, diagonal, coupling, cross, cubic = coef.T\\n    axis = np.arange(len(prev))\\n    def forward(z):\\n        p = z[:, prev]\\n        return bias + diagonal * z + coupling * p + cross * z * p + cubic * z ** 3\\n    linear = np.diag(diagonal)\\n    linear[axis, prev] = coupling\\n    affine = np.linalg.solve(linear, (x - bias).T).T\\n    use_warm = np.sum((forward(warm) - x) ** 2, axis=1) < np.sum((forward(affine) - x) ** 2, axis=1)\\n    z = np.where(use_warm[:, None], warm, affine)\\n    jac = np.zeros((len(z), len(axis), len(axis)))\\n    for _ in range(22):\\n        residual = forward(z) - x\\n        if np.max(np.abs(residual)) < 1e-11:\\n            break\\n        jac[:, axis, axis] = diagonal + cross * z[:, prev] + 3 * cubic * z * z\\n        jac[:, axis, prev] = coupling + cross * z\\n        try:\\n            step = np.linalg.solve(jac, residual[..., None])[..., 0]\\n        except np.linalg.LinAlgError:\\n            step = (np.linalg.pinv(jac) @ residual[..., None])[..., 0]\\n        best = np.sum(residual * residual, axis=1)\\n        new = z.copy()\\n        for damping in (1., .5, .25, .125, .0625, .03125):\\n            proposal = z - damping * step\\n            er = np.sum((forward(proposal) - x) ** 2, axis=1)\\n            better = er < best\\n            new[better], best[better] = proposal[better], er[better]\\n        z = new\\n    return z\\n\\n\\ndef _ridge(a, t, n, penalty):\\n    train = a[:n]\\n    reg = penalty * np.eye(a.shape[1])\\n    reg[0, 0] = 0.\\n    return a[n:] @ np.linalg.solve(train.T @ train + reg, train.T @ t)\\n\\n\\ndef _scalar(z, y, n, telemetry):\\n    h = z / np.sqrt(1 + telemetry[0] * z * z)\\n    p, q, r, s = h.T\\n    rat1 = q * r / (1 + (q + r) ** 2)\\n    rat2 = p * s / (1 + telemetry[2] * (p - s) ** 2)\\n    return _ridge(np.column_stack((_quadratic(h), np.sin(telemetry[2] * h), np.cos(telemetry[2] * h), rat1, rat2)), y, n, 1e-6)\\n\\n\\ndef _trajectory(z, traj, n):\\n    h = z / np.sqrt(1 + .09 * z * z)\\n    time = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    features = np.column_stack((_quadratic(h), np.sin(1.75 * h), np.cos(1.75 * h)))\\n    return _ridge(features, traj @ time.T, n, 1e-6) @ time\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study, frozen = inputs[\'__input__\'], inputs[\'node_00\']\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    x = np.row_stack((train[\'x\'], study[\'test\'][\'x\']))\\n    d = x.shape[1]\\n    start = np.asarray(frozen[\'hybrid\'])[:, -((d + 1) * (d + 2) * (d + 3) // 6):][:, 1:d + 1]\\n    coef, prev = _calibration(study[\'panel_a\'])\\n    z = _invert(x, start, coef, prev)\\n    y = _scalar(z, np.asarray(train[\'y\']).reshape(n), n, np.asarray(study[\'telemetry_vec\']))\\n    state = _trajectory(z, np.asarray(train[\'traj\']), n)\\n    diagnostics = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])\\n    return {\'y_hat\': y, \'state_pred\': state, \'diagnostics_vec\': diagnostics}\\n"}'

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
