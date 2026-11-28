"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_09", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "10cedbe8fbb1be5f328579cb2e700a3a5a45caa48ebe09e689b29ae0b7d9a35f", "node_01": "604721f044c02a31b0db65e5f9a6973b445f164412f5fe362b1fe126c195a833"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover the shared latent coordinates from the reference calibration.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    panel = study[\'panel_a\']\\n    u, v = np.asarray(panel[\'u\'], float), np.asarray(panel[\'v\'], float)\\n    d = u.shape[1]\\n    nxt = (np.arange(d) + 1) % d\\n\\n    def terms(q, k):\\n        return np.column_stack((np.ones(len(q)), q, q[:, k]**3, q[:, k]*q[:, nxt[k]]))\\n\\n    # Gross reference corruptions affect whole rows. Refine the joint inlier\\n    # mask after fitting the sparse cyclic model to the uncorrupted majority.\\n    delta = np.linalg.norm(v-u, axis=1)\\n    keep = delta <= max(3., 4.*np.median(delta))\\n    coef = np.stack([np.linalg.lstsq(terms(u[keep], k), v[keep,k], rcond=None)[0] for k in range(d)], axis=1)\\n\\n    def forward(q):\\n        return coef[0] + q @ coef[1:d+1] + coef[-2]*q**3 + coef[-1]*q*q[nxt]\\n\\n    errors = np.linalg.norm(np.array([forward(q) for q in u])-v, axis=1)\\n    good = errors <= max(0.003, 6.*np.median(errors))\\n    if np.count_nonzero(good) >= d+3:\\n        coef = np.stack([np.linalg.lstsq(terms(u[good], k), v[good,k], rcond=None)[0] for k in range(d)], axis=1)\\n\\n    def jac(q):\\n        out = coef[1:d+1].T.copy()\\n        k = np.arange(d)\\n        out[k,k] += 3.*coef[-2]*q*q + coef[-1]*q[nxt]\\n        out[k,nxt] += coef[-1]*q\\n        return out\\n\\n    train = np.asarray(study[\'train\'][\'x\'], float)\\n    test = np.asarray(study[\'test\'][\'x\'], float)\\n    x = np.concatenate((train, test), axis=0)\\n    init = np.linalg.lstsq(coef[1:d+1].T, (x-coef[0]).T, rcond=None)[0].T\\n    latent = np.empty_like(x)\\n    for i, (target, q) in enumerate(zip(x, init)):\\n        latent[i] = least_squares(lambda z: forward(z)-target, q, jac=jac, max_nfev=50,\\n                                  ftol=1e-11, xtol=1e-11, gtol=1e-11).x\\n    return {\'cubic_train\': latent[:len(train)]**3, \'cubic_test\': latent[len(train):]**3}\\n", "node_01": "\\"\\"\\"Calibrate the periodic panel operator and fit separate outcome readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _quadratic(x):\\n    i, j = np.triu_indices(x.shape[1])\\n    return np.column_stack((_affine(x), x[:, i] * x[:, j]))\\n\\n\\ndef _fit(a, y):\\n    return np.linalg.lstsq(a, y, rcond=None)[0]\\n\\n\\ndef _operator(panel, q):\\n    u = panel[\'u\']\\n    v = panel[\'v\']\\n\\n    def design(x, omega):\\n        return _affine(np.cos(omega * x))\\n\\n    def loss(omega):\\n        a = design(u, omega)\\n        return np.sum((a @ _fit(a, v) - v) ** 2)\\n\\n    omega = minimize_scalar(loss, bounds=(2.3, 2.9), method=\'bounded\',\\n                            options={\'xatol\': 1e-10}).x\\n    return design(q, omega) @ _fit(design(u, omega), v)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    latent = inputs[\'node_00\']\\n    train = study[\'train\']\\n    qt = np.cbrt(latent[\'cubic_train\'])\\n    qe = np.cbrt(latent[\'cubic_test\'])\\n    n = len(qt)\\n    b = _operator(study[\'panel_b\'], np.concatenate((qt, qe), axis=0))\\n\\n    d = _quadratic(b)\\n    y_hat = d[n:] @ _fit(d[:n], train[\'y\'])\\n\\n    a = _affine(b)\\n    traj = train[\'traj\']\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    state_pred = (a[n:] @ _fit(a[:n], traj @ modes.T)) @ modes\\n    end = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}\\n"}'

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
