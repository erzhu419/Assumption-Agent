"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 4, "fanout_by_node": {"node_00": 1, "node_01": 2, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_01"], "node_03": ["node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "215d9a9649fa524bd7edd62a2335215573c370f75de3a2ba9525eca64f1ca4b1", "node_01": "69c4d3f085eedea8d05dbcd97ffc59b8a7aeea5527819f2eb24ada5633ea56be", "node_02": "f9d37d1da43c0fe301660ed1cdcf195ba0d697d8fe1431e78060b5e508e99eaf", "node_03": "b6748aa005c40c6d864a932b98a4c28a1dc3a31eb8ffcc85dab661ae26874a4c"}, "shared_fanout_nodes": ["node_01"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates from the two paired reference panels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _a_design(u):\\n    nxt = np.roll(u, -1, axis=1)\\n    return np.stack((np.ones_like(u), u, nxt, u * nxt, u**3), axis=-1).transpose(1, 0, 2)\\n\\n\\ndef _fit_a(u, v):\\n    design = _a_design(u)\\n    target = v.T[..., None]\\n    rng = np.random.default_rng(485)\\n    best = (np.inf, None)\\n    for _ in range(128):\\n        idx = rng.choice(len(u), 8, replace=False)\\n        b = np.linalg.pinv(design[:, idx]) @ target[:, idx]\\n        residual = (design @ b - target)[..., 0]\\n        score = np.median(np.linalg.norm(residual, axis=0))\\n        if score < best[0]:\\n            best = (score, b)\\n    b = best[1]\\n    for _ in range(4):\\n        r = np.linalg.norm((design @ b - target)[..., 0], axis=0)\\n        keep = r <= max(6 * np.median(r), 1e-8)\\n        b = np.linalg.pinv(design[:, keep]) @ target[:, keep]\\n    return b[:, :, 0].T\\n\\n\\ndef _invert_a(x, b):\\n    x = np.asarray(x, dtype=float)\\n    bias, gain, cross, product, cubic = b\\n    d = x.shape[1]\\n    k = np.arange(d)\\n    nxt = (k + 1) % d\\n    linear = np.zeros((d, d))\\n    linear[k, k], linear[k, nxt] = gain, cross\\n    u = np.linalg.solve(linear, (x - bias).T).T\\n    for _ in range(35):\\n        v = u[:, nxt]\\n        r = bias + gain*u + cross*v + product*u*v + cubic*u**3 - x\\n        loss = np.max(np.abs(r), axis=1)\\n        if not len(loss) or np.max(loss) <= 1e-12:\\n            break\\n        jac = np.zeros((len(u), d, d))\\n        jac[:, k, k] = gain + product*v + 3*cubic*u*u\\n        jac[:, k, nxt] = cross + product*u\\n        delta = np.linalg.solve(jac, r[..., None])[..., 0]\\n        step = np.ones((len(u), 1))\\n        for _ in range(18):\\n            trial = u - step*delta\\n            v = trial[:, nxt]\\n            rr = bias + gain*trial + cross*v + product*trial*v + cubic*trial**3 - x\\n            bad = np.max(np.abs(rr), axis=1) > loss\\n            if not np.any(bad):\\n                break\\n            step[bad] *= 0.5\\n        u = np.where(bad[:, None], u, trial)\\n    return u\\n\\n\\ndef _b_design(u, scale):\\n    return np.column_stack((np.ones(len(u)), u, u*np.exp(-scale*u*u)))\\n\\n\\ndef _fit_b(u, v):\\n    def fit(scale):\\n        design = _b_design(u, scale)\\n        beta = np.linalg.lstsq(design, v, rcond=None)[0]\\n        return np.mean((design @ beta - v)**2), beta\\n    scale = minimize_scalar(lambda s: fit(s)[0], bounds=(0.1, 3.0), method=\'bounded\').x\\n    return scale, fit(scale)[1]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    a_fit = _fit_a(np.asarray(a[\'u\'], float), np.asarray(a[\'v\'], float))\\n    scale, beta = _fit_b(np.asarray(b[\'u\'], float), np.asarray(b[\'v\'], float))\\n    def coordinates(x):\\n        return _b_design(_invert_a(x, a_fit), scale) @ beta\\n    return {\'c\': coordinates(study[\'train\'][\'x\']),\\n            \'query\': coordinates(study[\'test\'][\'x\'])}\\n", "node_01": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _readout(c):\\n    return np.column_stack((_affine(c), np.sum(c[:, :2] * c[:, 2:], axis=1)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coords = inputs[\'node_00\']\\n    train = study[\'train\']\\n    c, query = coords[\'c\'], coords[\'query\']\\n    readout_beta = np.linalg.lstsq(_readout(c), np.asarray(train[\'y\'], dtype=float), rcond=None)[0]\\n    y_hat = _readout(query) @ readout_beta\\n    design, new_design = _affine(c), _affine(query)\\n    beta = np.linalg.lstsq(design, np.asarray(train[\'traj\'], dtype=float), rcond=None)[0]\\n    fitted = design @ beta\\n    times = np.asarray(study[\'eval_times\'], dtype=float)\\n    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)\\n    return {\'y_hat\': y_hat, \'beta\': beta, \'fitted\': fitted,\\n            \'new_design\': new_design, \'t\': times / scale}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    stage = inputs[\'node_01\']\\n    beta = np.asarray(stage[\'beta\'], dtype=float)\\n    fitted = np.asarray(stage[\'fitted\'], dtype=float)\\n    design = np.asarray(stage[\'new_design\'], dtype=float)\\n    _, _, modes = np.linalg.svd(fitted, full_matrices=False)\\n    basis = modes[:2].T\\n    return {\'state_pred\': (design @ (beta @ basis)) @ basis.T}\\n", "node_03": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    y_hat = inputs[\'node_01\'][\'y_hat\']\\n    state_pred = inputs[\'node_02\'][\'state_pred\']\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
