"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_12", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__"], "node_02": ["node_00", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "59ed851288afbdee0eb949ec0db8fca9294dd398d5638ff0d8fdafa4962e0fa9", "node_01": "1879b26e831195dba0cd078d4cfeb0456354c16168fc63c9ba65b58818a39a82", "node_02": "dea274ae6580bed80ee59e980fa6ff0119dcbe976fe71399de423ddb6bfaa61e"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Invert the calibration panel and construct the spatial interaction features.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _products(x, pairs=_PAIRS):\\n    return np.column_stack([x[:, i] * x[:, j] for i, j in pairs])\\n\\n\\ndef _coordinates(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    a = np.column_stack((_affine(u), u**3, _products(u)))\\n    keep = np.linalg.norm(v-u, axis=1) < 4.0\\n    if keep.sum() < a.shape[1]:\\n        keep = np.ones(len(u), dtype=bool)\\n    for _ in range(6):\\n        w = _fit(a[keep], v[keep])\\n        err = np.linalg.norm(a @ w - v, axis=1)\\n        new_keep = err <= max(1e-7, 6.0*np.median(err[keep]))\\n        if new_keep.sum() < a.shape[1] or np.array_equal(new_keep, keep):\\n            break\\n        keep = new_keep\\n    active = np.linalg.norm(w[9:], axis=1) > max(1e-3, 0.01*np.linalg.norm(w[9:], axis=1).max())\\n    pairs = tuple(p for p, b in zip(_PAIRS, active) if b)\\n    cols = np.r_[np.ones(9, dtype=bool), active]\\n    w = _fit(a[keep][:, cols], v[keep])\\n    lin, cub, cross = w[1:5], w[5:9], w[9:]\\n\\n    def value(q):\\n        z = w[0] + q @ lin + q**3 @ cub\\n        for (i, j), d in zip(pairs, cross):\\n            z += q[i]*q[j]*d\\n        return z\\n\\n    def jac(q):\\n        jmat = lin + 3*q[:, None]**2*cub\\n        for (i, j), d in zip(pairs, cross):\\n            jmat[i] += q[j]*d\\n            jmat[j] += q[i]*d\\n        return jmat.T\\n\\n    seeds = (x-w[0]) @ np.linalg.pinv(lin)\\n    return np.array([least_squares(lambda q: value(q)-obs, seed, jac=jac,\\n                                  ftol=1e-11, xtol=1e-11, gtol=1e-11).x\\n                     for obs, seed in zip(x, seeds)]).reshape(x.shape)\\n\\n\\ndef _operator(u, v, query, second):\\n    def base(x, k):\\n        if second:\\n            return np.column_stack((_affine(x), _products(x), x**3/(1+k*x*x)))\\n        return np.column_stack((_affine(x), x/(1+k*np.abs(x))))\\n\\n    def objective(k):\\n        a = base(u, k)\\n        return np.sum((a @ _fit(a, v)-v)**2)\\n\\n    grid = np.linspace(0.02, 3.0, 61)\\n    idx = int(np.argmin([objective(k) for k in grid]))\\n    k = minimize_scalar(objective, bounds=(grid[max(0, idx-1)], grid[min(idx+1, len(grid)-1)]),\\n                        method=\'bounded\', options={\'xatol\': 1e-9}).x\\n    if second:\\n        w = _fit(base(u, k), v)\\n        mags = np.linalg.norm(w[5:11], axis=1)\\n        active = mags > max(1e-3, 0.05*mags.max())\\n        cols = np.r_[np.ones(5, dtype=bool), active, np.ones(4, dtype=bool)]\\n        def reduced(k):\\n            a = base(u, k)[:, cols]\\n            return np.sum((a @ _fit(a, v)-v)**2)\\n        k = minimize_scalar(reduced, bounds=(max(0.001, k-0.2), k+0.2),\\n                            method=\'bounded\', options={\'xatol\': 1e-9}).x\\n    else:\\n        cols = slice(None)\\n    return base(query, k)[:, cols] @ _fit(base(u, k)[:, cols], v)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    x = np.concatenate((train[\'x\'], study[\'test\'][\'x\']))\\n    q = _coordinates(study[\'panel_a\'], x)\\n    panel = study[\'panel_c\']\\n    p = _operator(panel[\'u\'], panel[\'v1\'], q, False)\\n    r = _operator(panel[\'u\'], panel[\'v2\'], q, True)\\n    joint = p[:, :, None]*r[:, None, :]\\n    common = np.column_stack((p, r, np.trace(joint, axis1=1, axis2=2)))\\n    offdiag = joint[:, ~np.eye(joint.shape[1], dtype=bool)]\\n    f = _affine(common)\\n    y_hat = f[n:] @ _fit(f[:n], train[\'y\'])\\n    return {\'common\': common, \'offdiag\': offdiag, \'y_hat\': y_hat}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _temporal_basis(times, trajectories):\\n    t = np.asarray(times)\\n    def basis(rates):\\n        return -np.expm1(-t[:, None]*rates)\\n    def err(rates):\\n        f = basis(rates)\\n        return (f @ _ls(f, trajectories.T) - trajectories.T).ravel()\\n    opt = least_squares(err, [1.8, .43], bounds=([.8, .03], [8., .79]), max_nfev=80)\\n    return np.linalg.qr(basis(opt.x), mode=\'reduced\')[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    trajectories = study[\'train\'][\'traj\']\\n    basis = _temporal_basis(study[\'eval_times\'], trajectories)\\n    return {\'basis\': basis, \'scores\': trajectories @ basis}\\n", "node_02": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _select_interaction(common, offdiag, target):\\n    a = _affine(common)\\n    residual = target - a @ _ls(a, target)\\n    orthogonal = offdiag - a @ _ls(a, offdiag)\\n    energy = np.sum(orthogonal**2, axis=0)\\n    gain = np.sum((orthogonal.T @ residual)**2, axis=1) / np.maximum(energy, 1e-20)\\n    best = int(np.argmax(gain))\\n    return best if gain[best] > 0.15*np.sum(residual**2) else None\\n\\n\\ndef _ridge(f, target, query, penalty):\\n    fm, ym = f.mean(axis=0), target.mean(axis=0)\\n    a = f-fm\\n    w = np.linalg.solve(a.T @ a + penalty*np.eye(f.shape[1]), a.T @ (target-ym))\\n    return (query-fm) @ w + ym\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    spatial = inputs[\'node_00\']\\n    temporal = inputs[\'node_01\']\\n    common, offdiag, y = spatial[\'common\'], spatial[\'offdiag\'], spatial[\'y_hat\']\\n    basis, scores = temporal[\'basis\'], temporal[\'scores\']\\n    n = len(scores)\\n    j = _select_interaction(common[:n], offdiag[:n], scores)\\n    f = common if j is None else np.column_stack((common, offdiag[:, j]))\\n    state = _ridge(f[:n], scores, f[n:], 1e-6) @ basis.T\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}\\n"}'

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
