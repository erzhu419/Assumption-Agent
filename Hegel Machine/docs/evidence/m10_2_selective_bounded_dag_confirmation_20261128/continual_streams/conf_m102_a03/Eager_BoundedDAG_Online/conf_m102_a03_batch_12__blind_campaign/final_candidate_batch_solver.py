"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "40328e8a87244c972df80d4a393ed91ba48e3d12d9194c9db365029c180cc991", "node_01": "08b3e367d4b0c39350ec49aeebe12aa982177b0d46fa0c56b5b9a2bc1c891b3c"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(panel):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    a = _affine(u)\\n    coef = _fit(a, v)\\n    for _ in range(24):\\n        residual = np.linalg.norm(a @ coef - v, axis=1)\\n        scale = max(float(np.median(residual)), 1e-12)\\n        weight = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(residual, 1e-12)))\\n        coef = _fit(a * weight[:, None], v * weight[:, None])\\n    residual = np.linalg.norm(a @ coef - v, axis=1)\\n    good = residual <= 5 * max(float(np.median(residual)), 1e-12)\\n    if good.sum() >= a.shape[1] and np.linalg.matrix_rank(a[good]) == a.shape[1]:\\n        coef = _fit(a[good], v[good])\\n        mat = coef[1:]\\n        off = mat - np.diag(np.diag(mat))\\n        if np.linalg.norm(off) <= 5 * max(float(np.median(residual)), 1e-12):\\n            for j in range(u.shape[1]):\\n                coef[0, j], coef[j + 1, j] = _fit(a[good][:, [0, j + 1]], v[good, j])\\n            coef[1:] = np.diag(np.diag(coef[1:]))\\n    return coef\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.asarray(study[\'train\'][\'x\'], dtype=float)\\n    target = np.asarray(study[\'test\'][\'x\'], dtype=float)\\n    a = _calibration(study[\'panel_a\'])\\n    b = _fit(_affine(study[\'panel_b\'][\'u\']), study[\'panel_b\'][\'v\'])\\n\\n    def encode(rows):\\n        calibrated = np.linalg.solve(a[1:].T, (rows - a[0]).T).T\\n        return _affine(calibrated) @ b\\n\\n    return {\'z\': encode(x), \'zt\': encode(target)}\\n", "node_01": "\\"\\"\\"Correct the frozen affine alignment using the nonlinear reference panel.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _basis(x, frequency):\\n    phase = x * frequency\\n    return np.column_stack((_affine(x), np.sin(phase), np.cos(phase)))\\n\\n\\ndef _alignment(u, v):\\n    # Profile out the linear panel coefficients at each frequency update.\\n    frequency = np.full(u.shape[1], 2.9)\\n    ratio = (np.sqrt(5.) - 1.) / 2.\\n\\n    def loss(j, value):\\n        frequency[j] = value\\n        b = _basis(u, frequency)\\n        r = b @ _fit(b, v) - v\\n        return np.sum(r * r)\\n\\n    for _ in range(12):\\n        previous = frequency.copy()\\n        for j in range(len(frequency)):\\n            start = frequency[j]\\n            best = loss(j, start)\\n            lo, hi = 1.6, 4.2\\n            a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)\\n            fa, fb = loss(j, a), loss(j, b)\\n            for _ in range(22):\\n                if fa < fb:\\n                    hi, b, fb = b, a, fa\\n                    a = hi - ratio * (hi - lo)\\n                    fa = loss(j, a)\\n                else:\\n                    lo, a, fa = a, b, fb\\n                    b = lo + ratio * (hi - lo)\\n                    fb = loss(j, b)\\n            point, score = (a, fa) if fa < fb else (b, fb)\\n            frequency[j] = point if score < best else start\\n        if np.max(np.abs(frequency - previous)) < 1.e-5:\\n            break\\n    return frequency, _fit(_basis(u, frequency), v)\\n\\n\\ndef _readout(z):\\n    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]\\n    return np.column_stack((_affine(z), interaction))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    encoded = inputs[\'node_00\']\\n    u = np.asarray(study[\'panel_b\'][\'u\'], dtype=float)\\n    v = np.asarray(study[\'panel_b\'][\'v\'], dtype=float)\\n    affine = _fit(_affine(u), v)\\n    frequency, mapping = _alignment(u, v)\\n\\n    def correct(rows):\\n        raw = np.asarray(rows, dtype=float) - affine[0]\\n        x = np.linalg.solve(affine[1:].T, raw.T).T\\n        return _basis(x, frequency) @ mapping\\n\\n    z, zt = correct(encoded[\'z\']), correct(encoded[\'zt\'])\\n    train = study[\'train\']\\n    yh = _readout(zt) @ _fit(_readout(z), np.asarray(train[\'y\'], dtype=float))\\n    trajectory = np.asarray(train[\'traj\'], dtype=float)\\n    center = trajectory.mean(axis=0)\\n    _, _, vh = np.linalg.svd(trajectory - center, full_matrices=False)\\n    modes = vh[:2].T\\n    sp = center + (_affine(zt) @ _fit(_affine(z), (trajectory - center) @ modes)) @ modes.T\\n    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])\\n    return {\'y_hat\': yh, \'state_pred\': sp, \'diagnostics_vec\': diag}\\n"}'

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
