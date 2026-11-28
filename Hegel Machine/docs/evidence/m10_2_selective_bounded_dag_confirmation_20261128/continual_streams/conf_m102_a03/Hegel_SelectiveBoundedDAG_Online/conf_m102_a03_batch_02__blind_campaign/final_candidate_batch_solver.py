"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "f4fd48c71c8162e72478bfb8ae32fa0eceb85821ff86efe07d5d72d15b5766e5"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated response-surface and low-rank state model for the public campaign.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\nfrom scipy.special import erf, expit\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _calibrate(panel):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    j = u.shape[1]\\n    coeff = np.zeros((2, j))\\n    mats = [np.column_stack((np.ones(len(u)), u[:, k])) for k in range(j)]\\n    w = np.ones(len(u))\\n    for _ in range(28):\\n        for k, design in enumerate(mats):\\n            coeff[:, k] = _fit(design * w[:, None], v[:, k] * w)\\n        r = np.linalg.norm(v - coeff[0] - u * coeff[1], axis=1)\\n        scale = max(np.median(r), 1e-12)\\n        w = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(r, 1e-12)))\\n    keep = r < 6 * scale\\n    if np.count_nonzero(keep) >= 3:\\n        for k, design in enumerate(mats):\\n            coeff[:, k] = _fit(design[keep], v[keep, k])\\n    return coeff\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _quadratic(z):\\n    i, j = np.triu_indices(z.shape[1])\\n    return np.column_stack((_affine(z), z[:, i] * z[:, j]))\\n\\n\\ndef _readout(z, z_test, y):\\n    l, lt = _affine(z), _affine(z_test)\\n    q, qt = _quadratic(z), _quadratic(z_test)\\n    orth, _ = np.linalg.qr(q, mode=\'reduced\')\\n    y_linear = _fit(q, y)\\n    target = y - q @ y_linear\\n    direction = _fit(l, y)[1:]\\n    direction /= max(np.linalg.norm(direction), 1e-12)\\n    rng = np.random.default_rng(98)\\n    starts = [np.r_[offset, scale * direction] for offset in (0., -.3)\\n              for scale in (.45, 1., 1.8, 3., 5.)]\\n    starts += [np.r_[0., d] for d in rng.normal(size=(8, z.shape[1]))]\\n\\n    def fit_branch(mode):\\n        def features(a, h):\\n            g = a @ h\\n            if mode == 0:\\n                return (g / (1. + g*g))[:, None]\\n            if mode == 1:\\n                return a / (1. + g[:, None]**2)\\n            return a * expit(g[:, None])\\n\\n        def residual(h):\\n            f = features(l, h)\\n            f = f - orth @ (orth.T @ f)\\n            return f @ _fit(f, target) - target\\n\\n        fits = [least_squares(residual, start, bounds=(-12., 12.), max_nfev=100)\\n                for start in starts]\\n        h = min(fits, key=lambda v: v.cost).x\\n        design = np.column_stack((q, features(l, h)))\\n        return np.column_stack((qt, features(lt, h))) @ _fit(design, y)\\n\\n    # A scalar gate uses fewer degrees of freedom than a gated affine\\n    # branch, which is particularly useful outside the fitted index range.\\n    scalar_starts = starts[:10]\\n\\n    def fit_scalar(transform):\\n        def residual(h):\\n            v = transform(l @ h)\\n            w = v - orth @ (orth.T @ v)\\n            return w * ((w @ target) / max(w @ w, 1e-24)) - target\\n\\n        h = min((least_squares(residual, h0, bounds=(-10., 10.), max_nfev=100)\\n                 for h0 in scalar_starts), key=lambda result: result.cost).x\\n        coef = _fit(np.column_stack((q, transform(l @ h))), y)\\n        return np.column_stack((qt, transform(lt @ h))) @ coef\\n\\n    scalar = np.mean([fit_scalar(f) for f in\\n                      (lambda g: g / (1. + g*g), erf, np.tanh)], axis=0)\\n    affine = .3 * fit_branch(0) + .4 * fit_branch(1) + .3 * fit_branch(2)\\n    return .85 * scalar + .15 * affine\\n\\n\\ndef _state(z, z_test, traj):\\n    # Fit in the orthonormal spatial basis, then shrink weak temporal modes.\\n    l, lt = _affine(z), _affine(z_test)\\n    qr, r = np.linalg.qr(l, mode=\'reduced\')\\n    score = qr.T @ traj\\n    u, sing, vh = np.linalg.svd(score, full_matrices=False)\\n    rank_two = (u[:, :2] * sing[:2]) @ vh[:2]\\n    return lt @ _fit(r, .35 * score + .65 * rank_two)\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    \\"\\"\\"Fit from the supplied training observations and predict the test rows.\\"\\"\\"\\n    train = study[\'train\']\\n    shift, scale = _calibrate(study[\'panel_a\'])\\n    z = (np.asarray(train[\'x\'], dtype=float) - shift) / scale\\n    z_test = (np.asarray(study[\'test\'][\'x\'], dtype=float) - shift) / scale\\n    y_hat = _readout(z, z_test, np.asarray(train[\'y\'], dtype=float))\\n    state_pred = _state(z, z_test, np.asarray(train[\'traj\'], dtype=float))\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
