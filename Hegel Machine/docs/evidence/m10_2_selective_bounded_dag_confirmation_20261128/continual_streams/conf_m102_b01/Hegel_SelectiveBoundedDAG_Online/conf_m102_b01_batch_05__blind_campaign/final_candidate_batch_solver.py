"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "5dc93da6397c1829c962dc50f63d13769284e7153c151d05b4a76fcad5c0e482", "node_01": "fb16ab30b10dc2b8bff66984afc0f44457757bd1acb436bed387ac33dffdb0a6"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _a(u, v, x):\\n    j = np.roll(np.arange(u.shape[1]), -1)\\n    def basis(t, z):\\n        return np.stack([np.ones_like(t), t, z, t*z, t**3], axis=-1)\\n    f = basis(u, u[:, j])\\n    keep = np.max(np.abs(v-u), axis=1) < 3\\n    coef = np.stack([_fit(f[keep, i], v[keep, i]) for i in range(u.shape[1])])\\n    return np.stack([least_squares(lambda z: (basis(z, z[j])*coef).sum(1)-row, row, gtol=1e-11).x for row in x])\\n\\n\\ndef _b(u, v, x):\\n    # A shared odd response acts on each coordinate before affine mixing.\\n    # Profile out the mixing and offset using only the calibration pairs.\\n    def basis(z, p):\\n        curvature, linear, correction, width = p\\n        z2 = z*z\\n        response = (z - z/np.sqrt(1 + curvature*z2)\\n                    + linear*z + correction*z/(1 + width*z2))\\n        return np.column_stack([np.ones(len(z)), response])\\n\\n    def residual(p):\\n        f = basis(u, p)\\n        return (f @ _fit(f, v) - v).ravel()\\n\\n    p = least_squares(residual, [1., 0., .1, 15.],\\n                      bounds=([.02, -1., -1., 1.], [10., 1., 1., 100.]),\\n                      max_nfev=400).x\\n    return basis(x, p) @ _fit(basis(u, p), v)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    x = np.concatenate([study[\'train\'][\'x\'], study[\'test\'][\'x\']])\\n    return {\'w\': _b(b[\'u\'], b[\'v\'], _a(a[\'u\'], a[\'v\'], x))}\\n", "node_01": "from itertools import combinations_with_replacement\\nimport numpy as np\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _trajectory_terms(x, y):\\n    n = len(y)\\n    terms = list(range(x.shape[1]))\\n\\n    def criterion(cols):\\n        a = x[:, cols]\\n        residual = y - a @ _fit(a, y)\\n        return n * np.log(max(np.mean(residual**2), 1e-30)) + len(cols) * np.log(n)\\n\\n    while len(terms) > 1:\\n        value, drop = min((criterion([j for j in terms if j != i]), i) for i in terms)\\n        if value >= criterion(terms):\\n            break\\n        terms.remove(drop)\\n    return terms\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    w = np.asarray(inputs[\'node_00\'][\'w\'], dtype=float)\\n    y = np.asarray(train[\'y\'], dtype=float)\\n    trajectory = np.asarray(train[\'traj\'], dtype=float)\\n    pairs = list(combinations_with_replacement(range(w.shape[1]), 2))\\n    products = [w[:, i] * w[:, j] for i, j in pairs]\\n    features = np.column_stack([np.ones(len(w)), w, *products])\\n    n = len(y)\\n    a, b = features[:n], features[n:]\\n    terms = _trajectory_terms(a, trajectory)\\n    # The scalar head is quadratic in the active bilinear coordinates,\\n    # not a general fourth-degree polynomial in the original coordinates.\\n    offset = 1 + w.shape[1]\\n    active = [j for j in terms if j >= offset and\\n              pairs[j - offset][0] != pairs[j - offset][1]]\\n    k = len(active)\\n    if k and 1 + k + k * (k + 1) // 2 + 5 <= n:\\n        scalar = np.column_stack([np.ones(len(w)), features[:, active],\\n                                  *[features[:, i] * features[:, j]\\n                                    for i, j in combinations_with_replacement(active, 2)]])\\n        y_hat = scalar[n:] @ _fit(scalar[:n], y)\\n    else:\\n        y_hat = b @ _fit(a, y)\\n    state_pred = b[:, terms] @ _fit(a[:, terms], trajectory)\\n    tail = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), tail.mean(), tail.std()])}\\n"}'

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
