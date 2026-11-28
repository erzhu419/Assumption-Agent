"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__"], "node_02": ["node_00", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "fd0368409891c083bab58a9bdb2a9f4f81c50016e03a41d49a090e16cfbc2d00", "node_01": "7071c5a1e3b4cd1f25faf290dc6eb49e752232ccd02032e60c345b25cdd8b2f4", "node_02": "c13e3728d66776168e0f0db4116d78855b81fdceabf9ce621fe4ce10cc0a02eb"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates from two paired calibration panels.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _powers(d):\\n    return [()] + [(i,) for i in range(d)] + [p for k in (2, 3) for p in combinations_with_replacement(range(d), k)]\\n\\n\\ndef _design(x, powers):\\n    return np.stack([np.prod(x[..., p], axis=-1) if p else np.ones(x.shape[:-1]) for p in powers], axis=-1)\\n\\n\\ndef _calibrate(u, v):\\n    powers = _powers(u.shape[1])\\n    a = _design(u, powers[:1+u.shape[1]])\\n    keep = np.ones(len(u), dtype=bool)\\n    for _ in range(6):\\n        w = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]\\n        r = np.linalg.norm(a @ w - v, axis=1)\\n        proposed = r < max(0.003, 4*np.median(r[keep]))\\n        if proposed.sum() < len(powers) or np.array_equal(proposed, keep):\\n            break\\n        keep = proposed\\n    a = _design(u[keep], powers)\\n    w = np.linalg.lstsq(a, v[keep], rcond=None)[0]\\n    # Refit after dropping coefficients compatible with reference noise.\\n    mask = np.max(np.abs(w), axis=1) > 0.002\\n    mask[:1+u.shape[1]] = True\\n    return [p for p, m in zip(powers, mask) if m], np.linalg.lstsq(a[:, mask], v[keep], rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    powers, w = _calibrate(np.asarray(a[\'u\'], float), np.asarray(a[\'v\'], float))\\n    x, xt = (np.asarray(study[k][\'x\'], float) for k in (\'train\', \'test\'))\\n    observations = np.concatenate((x, xt))\\n    d = x.shape[1]\\n    initial = np.linalg.solve(w[1:d+1].T, (observations - w[0]).T).T\\n    latent = np.empty_like(observations)\\n    for i, (v, u0) in enumerate(zip(observations, initial)):\\n        latent[i] = least_squares(lambda u: _design(u, powers) @ w - v, u0, gtol=1e-12, max_nfev=60).x\\n    ap = _powers(d)[:d+1]\\n    wb = np.linalg.lstsq(_design(np.asarray(b[\'u\'], float), ap), np.asarray(b[\'v\'], float), rcond=None)[0]\\n    z = _design(latent, ap) @ wb\\n    return {\'z\': z[:len(x)].copy(), \'zt\': z[len(x):].copy()}\\n", "node_01": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    traj = np.asarray(train[\'traj\'], dtype=float)\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    targets = np.column_stack((train[\'y\'], traj @ modes.T))\\n    return {\'modes\': modes, \'targets\': targets}\\n", "node_02": "import numpy as np\\n\\n\\ndef _kernel(z, w):\\n    h, g = np.arcsinh(3.5 * z), np.arcsinh(3.5 * w)\\n    linear = z @ w.T\\n    compressed = h @ g.T\\n    return (1 + linear + compressed + (h*h) @ (g*g).T\\n            + (z*h) @ (w*g).T + 0.2 * (compressed + 0.25*linear)**2)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    coordinates = inputs[\'node_00\']\\n    projected = inputs[\'node_01\']\\n    z, zt = coordinates[\'z\'], coordinates[\'zt\']\\n    modes, targets = projected[\'modes\'], projected[\'targets\']\\n    gram = _kernel(z, z)\\n    gram.flat[::len(z)+1] += 1e-6 * np.mean(np.diag(gram))\\n    pred = _kernel(zt, z) @ np.linalg.solve(gram, targets)\\n    y = pred[:, 0].copy()\\n    state = pred[:, 1:] @ modes\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
