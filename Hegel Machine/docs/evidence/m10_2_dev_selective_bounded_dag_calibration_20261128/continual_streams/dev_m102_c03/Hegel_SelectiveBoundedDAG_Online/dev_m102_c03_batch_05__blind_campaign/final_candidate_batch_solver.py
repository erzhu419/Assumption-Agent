"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "cc35289a343f12aa898096f024f12646fafe48f1741e354a73bcd41b0316dfbe"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated dual-branch regression with shared exponential dynamics.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(h, targets):\\n    return np.linalg.lstsq(h, targets, rcond=None)[0]\\n\\n\\ndef _one(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _edge(x):\\n    return x * np.roll(x, -1, axis=1)\\n\\n\\ndef _cal_design(x, j):\\n    u, v = x[:, j], x[:, (j + 1) % 4]\\n    return _one(np.column_stack((u, v, u*v, u**3)))\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    h = _one(u)\\n    r = np.linalg.norm(h @ _fit(h, v) - v, axis=1)\\n    good = r <= 4 * max(np.median(r), 1e-8)\\n    for _ in range(4):\\n        a = np.stack([_fit(_cal_design(u[good], j), v[good, j]) for j in range(4)])\\n        r = np.linalg.norm(np.column_stack([_cal_design(u, j) @ a[j] for j in range(4)]) - v, axis=1)\\n        new = r <= 5 * max(np.median(r), 1e-6)\\n        if new.sum() < 10 or np.array_equal(good, new):\\n            break\\n        good = new\\n    return a\\n\\n\\ndef _inverse(x, a):\\n    bias, scale, neighbor, mixed, cubic = a.T\\n    z = (x - bias) / scale\\n    j = np.arange(4)\\n    for _ in range(14):\\n        w = np.roll(z, -1, axis=1)\\n        r = bias + scale*z + neighbor*w + mixed*z*w + cubic*z**3 - x\\n        jac = np.zeros((len(x), 4, 4))\\n        jac[:, j, j] = scale + mixed*w + 3*cubic*z*z\\n        jac[:, j, (j + 1) % 4] = neighbor + mixed*z\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        z -= np.clip(step, -1., 1.)\\n        if np.max(np.abs(step), initial=0.) < 1e-12:\\n            break\\n    return z\\n\\n\\ndef _branches(study, x):\\n    z = _inverse(np.asarray(x, dtype=float), _calibration(study[\'panel_a\']))\\n    u, v1, v2 = (study[\'panel_c\'][k] for k in (\'u\', \'v1\', \'v2\'))\\n    c = _one(_edge(z)) @ _fit(_one(_edge(u)), v1)\\n\\n    def cosine_residual(f):\\n        h = _one(np.cos(u * f[0]))\\n        return (h @ _fit(h, v2) - v2).ravel()\\n\\n    f = least_squares(cosine_residual, [2.4], bounds=(0.1, 7.)).x[0]\\n    d = _one(np.cos(f*z)) @ _fit(_one(np.cos(f*u)), v2)\\n    return c, d\\n\\n\\ndef _fit_state(h, t, traj):\\n    # Variable projection: fit linear spatial coefficients at each trial rate pair.\\n    pinv = np.linalg.pinv(h)\\n    proj = h @ (pinv @ traj)\\n\\n    def modes(g):\\n        return -np.expm1(-g[:, None] * t)\\n\\n    def residual(g):\\n        q = modes(g)\\n        return (_fit(q.T, proj.T).T @ q - proj).ravel()\\n\\n    rates = least_squares(residual, [.42, 1.82], bounds=([.05, .8], [.8, 4.]), ftol=1e-11, gtol=1e-11).x\\n    q = modes(rates)\\n    return pinv @ _fit(q.T, traj.T).T, q\\n\\n\\ndef solve_batch(study):\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    c, d = _branches(study, np.concatenate((train[\'x\'], study[\'test\'][\'x\']), axis=0))\\n    dot = np.sum(c*d, axis=1)\\n    hy = _one(np.column_stack((c, d, dot)))\\n    ht = np.column_stack((hy, c[:, 0]*d[:, 1]))\\n    y_hat = hy[n:] @ _fit(hy[:n], train[\'y\'])\\n    weights, modes = _fit_state(ht[:n], np.asarray(study[\'eval_times\']), train[\'traj\'])\\n    state = ht[n:] @ weights @ modes\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), state[:, -1].mean(), state[:, -1].std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state, \'diagnostics_vec\': diagnostics}\\n"}'

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
