"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["__input__", "node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "ea88124686cd2e8004ca01a18a89261e79bc9e041b39cf480913fc22a10eba59", "node_01": "819cd452e3319301325342006b8aa89dcdf371265b9518e87128f14919d9da13", "node_02": "61e1a4803452c5315639b459b5ecef9edf87fd6af8f74a1977a915de0fcac27b"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n\\n    def block(a, i):\\n        p, q = a[:, i], a[:, (i + 1) % 4]\\n        return np.column_stack((np.ones(len(a)), p, q, p*q, p**3))\\n\\n    blocks = [block(u, i) for i in range(4)]\\n    mask = np.ones(len(u), dtype=bool)\\n    for _ in range(8):\\n        coef = [_fit(a[mask], v[mask, i]) for i, a in enumerate(blocks)]\\n        err = np.linalg.norm(v - np.column_stack([a @ c for a, c in zip(blocks, coef)]), axis=1)\\n        med = np.median(err[mask])\\n        mad = np.median(abs(err[mask] - med))\\n        updated = err <= max(0.002, med + 5 * 1.4826 * mad)\\n        if updated.sum() < 12 or np.array_equal(updated, mask):\\n            break\\n        mask = updated\\n    c = np.stack([_fit(a[mask], v[mask, i]) for i, a in enumerate(blocks)])\\n\\n    def forward(a):\\n        b = np.roll(a, -1)\\n        return c[:, 0] + c[:, 1]*a + c[:, 2]*b + c[:, 3]*a*b + c[:, 4]*a**3\\n\\n    def jac(a):\\n        i = np.arange(4)\\n        j = np.diag(c[:, 1] + c[:, 3]*np.roll(a, -1) + 3*c[:, 4]*a*a)\\n        j[i, (i+1)%4] = c[:, 2] + c[:, 3]*a\\n        return j\\n\\n    return np.asarray([least_squares(lambda a: forward(a) - b, b, jac=jac, max_nfev=60).x for b in x]).reshape(-1, 4)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.asarray(study[\'train\'][\'x\'], float)\\n    test = np.asarray(study[\'test\'][\'x\'], float)\\n    z = _calibrate(study[\'panel_a\'], np.concatenate((x, test)))\\n    return {\'z\': z, \'train_idx\': np.arange(len(x)), \'test_idx\': np.arange(len(x), len(z))}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\nfrom scipy.special import erf\\n\\n\\ndef _aff(a):\\n    return np.column_stack((np.ones(len(a)), a))\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    panel = inputs[\'__input__\'][\'panel_b\']\\n    z = inputs[\'node_00\'][\'z\']\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n\\n    def design(a, w):\\n        return np.column_stack((_aff(a), erf(w*a)))\\n\\n    def loss(w):\\n        a = design(u, w)\\n        return np.mean((a @ _fit(a, v) - v)**2)\\n\\n    w = minimize_scalar(loss, bounds=(0.4, 3.0), method=\'bounded\').x\\n    return {\'g\': design(z, w) @ _fit(design(u, w), v)}\\n", "node_02": "\\"\\"\\"Separate calibrated endpoint and low-rank trajectory readouts.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\nimport numpy as np\\nfrom sklearn.linear_model import Lasso\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=1e-10)[0]\\n\\n\\ndef _structures(z, g):\\n    cyclic = z * np.roll(g, 1, axis=1)\\n    affine = np.column_stack((np.ones(len(z)), g, z, cyclic))\\n    core = np.column_stack((affine, g[:, :2] * g[:, 2:]))\\n    values = np.column_stack((z, g))\\n    terms = []\\n    for degree in (2, 3):\\n        for inds in combinations_with_replacement(range(values.shape[1]), degree):\\n            terms.append(np.prod(values[:, inds], axis=1))\\n    for i in range(values.shape[1]):\\n        for j in range(i):\\n            for sign in (-1, 1):\\n                h = values[:, i] + sign * values[:, j]\\n                for weight in (0.5, 1.0, 2.0):\\n                    x = weight * h\\n                    terms.extend((np.sin(x), np.cos(x), np.tanh(x), x / (1 + x*x)))\\n    for x in values.T:\\n        for t in values.T:\\n            for weight in (0.5, 1.0, 2.0):\\n                terms.append(x / (1 + weight * t*t))\\n    return affine, core, np.column_stack(terms)\\n\\n\\ndef _endpoint(core, terms, train_idx, test_idx, y):\\n    fit = _fit(core[train_idx], np.column_stack((y, terms[train_idx])))\\n    target = y - core[train_idx] @ fit[:, 0]\\n    residual = terms - core @ fit[:, 1:]\\n    scale = np.maximum(residual[train_idx].std(axis=0), 1e-6)\\n    model = Lasso(alpha=0.001, max_iter=6000, tol=0.0003, fit_intercept=False)\\n    model.fit(residual[train_idx] / scale, target)\\n    return core[test_idx] @ fit[:, 0] + model.predict(residual[test_idx] / scale)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    observed = inputs[\'__input__\'][\'train\']\\n    calibrated = inputs[\'node_00\']\\n    z = np.asarray(calibrated[\'z\'], float)\\n    i, j = calibrated[\'train_idx\'], calibrated[\'test_idx\']\\n    g = np.asarray(inputs[\'node_01\'][\'g\'], float)\\n    y = np.asarray(observed[\'y\'], float).reshape(-1)\\n    traj = np.asarray(observed[\'traj\'], float)\\n    affine, core, terms = _structures(z, g)\\n    yp = _endpoint(core, terms, i, j, y)\\n    _, _, vt = np.linalg.svd(traj, full_matrices=False)\\n    modes = vt[:3]\\n    tp = (affine[j] @ _fit(affine[i], traj @ modes.T)) @ modes\\n    diagnostic = np.array([yp.mean(), yp.std(), tp[:, -1].mean(), tp[:, -1].std()])\\n    return {\'y_hat\': yp, \'state_pred\': tp, \'diagnostics_vec\': diagnostic}\\n"}'

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
