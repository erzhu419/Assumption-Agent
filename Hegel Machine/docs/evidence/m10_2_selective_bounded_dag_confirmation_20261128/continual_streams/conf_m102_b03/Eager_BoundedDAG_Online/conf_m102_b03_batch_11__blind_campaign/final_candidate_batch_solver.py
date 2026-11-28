"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "80950fa81a8ec8086e232c881626f02ef2981e51cff6c5c6b42d4e187e1fc6a7"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Fit a paired nonlinear calibration, then predict dual-branch responses.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ols(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _design(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _soft(z):\\n    return z / (1.0 + np.abs(z))\\n\\n\\ndef _calibration(panel):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    affine = _design(u)\\n    keep = np.arange(len(u))\\n    for _ in range(6):\\n        linear = _ols(affine[keep], v[keep])\\n        err = np.linalg.norm(affine @ linear - v, axis=1)\\n        med = np.median(err)\\n        mad = np.median(np.abs(err - med))\\n        nxt = np.flatnonzero(err <= max(0.015, med + 7.0 * 1.4826 * mad))\\n        if len(nxt) < 12 or np.array_equal(nxt, keep):\\n            break\\n        keep = nxt\\n    ug, vg = u[keep], v[keep]\\n\\n    def design(z, j, shape):\\n        a, b = z[:, j], z[:, (j + 2) % 4]\\n        return _design(np.column_stack((a, _soft(shape[0] * a), b, np.tanh(shape[1] * b))))\\n\\n    def residual(shape):\\n        residuals = []\\n        for j in range(4):\\n            d = design(ug, j, shape)\\n            residuals.append(d @ _ols(d, vg[:, j]) - vg[:, j])\\n        return np.concatenate(residuals)\\n\\n    shape = least_squares(residual, [0.4, 1.0], bounds=(0.08, 8.0), gtol=1e-12).x\\n    coef = np.column_stack([_ols(design(ug, j, shape), vg[:, j]) for j in range(4)])\\n    a, b = shape\\n    col = np.arange(4)\\n    opp = (col + 2) % 4\\n\\n    def inverse(x):\\n        x = np.asarray(x, dtype=float)\\n        z = (x - linear[0]) @ np.linalg.pinv(linear[1:])\\n\\n        def model(z):\\n            w = z[..., opp]\\n            return coef[0] + coef[1] * z + coef[2] * _soft(a * z) + coef[3] * w + coef[4] * np.tanh(b * w)\\n\\n        def jac(z):\\n            j = np.zeros(z.shape[:-1] + (4, 4))\\n            j[..., col, col] = coef[1] + coef[2] * a / (1.0 + np.abs(a * z))**2\\n            j[..., col, opp] = coef[3] + coef[4] * b * (1.0 - np.tanh(b * z[..., opp])**2)\\n            return j\\n\\n        for _ in range(16):\\n            r = model(z) - x\\n            if not len(z) or np.max(np.abs(r)) <= 1e-11:\\n                break\\n            step = (np.linalg.pinv(jac(z)) @ r[..., None])[..., 0]\\n            z -= np.clip(step, -0.5, 0.5)\\n        for i in np.flatnonzero(np.max(np.abs(model(z) - x), axis=1) > 1e-7):\\n            z[i] = least_squares(lambda p: model(p) - x[i], z[i], jac=jac).x\\n        return z\\n\\n    return inverse\\n\\n\\ndef _branches(panel):\\n    u = np.asarray(panel[\'u\'], dtype=float)\\n\\n    def fit(v, transform, start):\\n        def design(z, scale):\\n            return _design(np.column_stack((z, transform(scale[0] * z))))\\n\\n        def residual(scale):\\n            d = design(u, scale)\\n            return (d @ _ols(d, v) - v).ravel()\\n\\n        scale = least_squares(residual, [start], bounds=(0.05, 4.0), gtol=1e-13).x\\n        coef = _ols(design(u, scale), v)\\n        return lambda z: design(z, scale) @ coef\\n\\n    return (fit(panel[\'v1\'], _soft, 0.84),\\n            fit(panel[\'v2\'], lambda z: 1.0 / (1.0 + z*z), 0.4))\\n\\n\\ndef _features(study, x):\\n    z = _calibration(study[\'panel_a\'])(x)\\n    branch1, branch2 = _branches(study[\'panel_c\'])\\n    c, d = branch1(z), branch2(z)\\n    outer = (c[:, :, None] * d[:, None, :]).reshape(len(x), 16)\\n    base = np.column_stack((c, d))\\n    return (_design(np.column_stack((base, np.sum(c*d, axis=1)))),\\n            _design(np.column_stack((base, outer))))\\n\\n\\ndef _trajectory(t, observed, train_features, test_features):\\n    def basis(rates):\\n        return -np.expm1(-t[:, None] * rates)\\n\\n    def residual(rates):\\n        h = basis(rates)\\n        return (h @ _ols(h, observed.T) - observed.T).ravel()\\n\\n    rates = least_squares(residual, [1.8, 0.4], bounds=([0.65, 0.025], [6.0, 0.65]), gtol=1e-11).x\\n    h = basis(rates)\\n    amplitudes = _ols(h, observed.T).T\\n    return (test_features @ _ols(train_features, amplitudes)) @ h.T\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    \\"\\"\\"Return scalar predictions, time-grid predictions, and population moments.\\"\\"\\"\\n    tr = study[\'train\']\\n    n = len(tr[\'x\'])\\n    fy, ft = _features(study, np.vstack((tr[\'x\'], study[\'test\'][\'x\'])))\\n    y = fy[n:] @ _ols(fy[:n], np.asarray(tr[\'y\']).reshape(n))\\n    state = _trajectory(np.asarray(study[\'eval_times\']), np.asarray(tr[\'traj\']), ft[:n], ft[n:])\\n    terminal = state[:, -1]\\n    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()]) if len(y) else np.zeros(4)\\n    return {\'y_hat\': y, \'state_pred\': state, \'diagnostics_vec\': diagnostics}\\n"}'

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
