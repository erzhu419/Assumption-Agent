"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "0fcbea489f6ffab08b75ec234fa991f770dc411ad6e9272f616d87c87d87c95f"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated cyclic operator with separate scalar and temporal readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\nfrom scipy.special import erf\\n\\n\\ndef _aff(a):\\n    return np.column_stack((np.ones(len(a)), a))\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n    # Each reference coordinate couples to its successor on the four-cycle.\\n    def block(a, i):\\n        p, q = a[:, i], a[:, (i + 1) % 4]\\n        return np.column_stack((np.ones(len(a)), p, q, p*q, p**3))\\n    blocks = [block(u, i) for i in range(4)]\\n    mask = np.ones(len(u), dtype=bool)\\n    for _ in range(8):\\n        coef = [_fit(a[mask], v[mask, i]) for i, a in enumerate(blocks)]\\n        err = np.linalg.norm(v - np.column_stack([a @ c for a, c in zip(blocks, coef)]), axis=1)\\n        med = np.median(err[mask]); mad = np.median(abs(err[mask] - med))\\n        updated = err <= max(0.002, med + 5 * 1.4826 * mad)\\n        if updated.sum() < 12 or np.array_equal(updated, mask):\\n            break\\n        mask = updated\\n    c = np.stack([_fit(a[mask], v[mask, i]) for i, a in enumerate(blocks)])\\n    def forward(a):\\n        b = np.roll(a, -1)\\n        return c[:, 0] + c[:, 1]*a + c[:, 2]*b + c[:, 3]*a*b + c[:, 4]*a**3\\n    def jac(a):\\n        i = np.arange(4)\\n        j = np.diag(c[:, 1] + c[:, 3]*np.roll(a, -1) + 3*c[:, 4]*a*a)\\n        j[i, (i+1)%4] = c[:, 2] + c[:, 3]*a\\n        return j\\n    return np.asarray([least_squares(lambda a: forward(a) - b, b, jac=jac, max_nfev=60).x for b in x]).reshape(-1, 4)\\n\\n\\ndef _panel(panel, z):\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n    def design(a, w):\\n        return np.column_stack((_aff(a), erf(w*a)))\\n    def loss(w):\\n        a = design(u, w)\\n        return np.mean((a @ _fit(a, v) - v)**2)\\n    w = minimize_scalar(loss, bounds=(0.4, 3.0), method=\'bounded\').x\\n    return design(z, w) @ _fit(design(u, w), v)\\n\\n\\ndef _predict(z, g, train_idx, test_idx, y, traj):\\n    v = z * np.roll(g, 1, axis=1)\\n    a = _aff(np.column_stack((g, z, v)))\\n    q = g[:, :2] * g[:, 2:]\\n    b = np.column_stack((a, q))\\n    # Shrink supplemental terms using their scale after projecting out the core.\\n    extras = np.column_stack((z*g, z*np.roll(v, 2, axis=1), g*np.roll(g, 1, axis=1)))\\n    core = _fit(b[train_idx], np.column_stack((y, extras[train_idx])))\\n    r = y - b[train_idx] @ core[:, 0]\\n    e = extras - b @ core[:, 1:]\\n    scl = np.maximum(np.linalg.norm(e[train_idx], axis=0), 1e-8)\\n    e = e / scl\\n    w = np.linalg.solve(e[train_idx].T @ e[train_idx] + 2.0*np.eye(e.shape[1]), e[train_idx].T @ r)\\n    yp = b[test_idx] @ core[:, 0] + e[test_idx] @ w\\n    # State observations share smooth temporal modes; keep the dominant three.\\n    _, _, vt = np.linalg.svd(traj, full_matrices=False)\\n    modes = vt[:3]\\n    tp = (a[test_idx] @ _fit(a[train_idx], traj @ modes.T)) @ modes\\n    return yp, tp\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    x = np.asarray(train[\'x\'], float)\\n    test = np.asarray(study[\'test\'][\'x\'], float)\\n    z = _calibrate(study[\'panel_a\'], np.concatenate((x, test)))\\n    g = _panel(study[\'panel_b\'], z)\\n    y, t = _predict(z, g, np.arange(len(x)), np.arange(len(x), len(z)), np.asarray(train[\'y\']), np.asarray(train[\'traj\']))\\n    diagnostic = np.array([y.mean(), y.std(), t[:, -1].mean(), t[:, -1].std()])\\n    return {\'y_hat\': y, \'state_pred\': t, \'diagnostics_vec\': diagnostic}\\n"}'

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
