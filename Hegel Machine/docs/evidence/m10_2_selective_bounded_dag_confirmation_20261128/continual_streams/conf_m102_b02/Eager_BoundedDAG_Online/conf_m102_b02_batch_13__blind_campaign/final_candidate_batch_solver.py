"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_19", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "4353063e1a3fb27dc30b7a76314daa023116c94bb959582a750077b625608f24", "node_01": "40d18571ae358b875c90f50f4c59694ae6a468cd9a5f81f7146e9452b72deeb0", "node_02": "d94d6bbf564efb7d9f3cfe9cd7fc376b592b0c235d5a682ced6b560fe99778fb"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates from calibration panels for the frozen kernel.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _design(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(a, v):\\n    return np.linalg.lstsq(a, v, rcond=1e-10)[0]\\n\\n\\ndef _latent(panel, x):\\n    a, v = _design(panel[\'u\']), np.asarray(panel[\'v\'])\\n    keep = np.ones(len(a), dtype=bool)\\n    for _ in range(8):\\n        w = _fit(a[keep], v[keep])\\n        e = np.linalg.norm(a @ w - v, axis=1)\\n        good = e < max(.001, 3 * np.median(e))\\n        if np.array_equal(keep, good) or np.count_nonzero(good) < a.shape[1]:\\n            break\\n        keep = good\\n    return np.linalg.solve(w[1:].T, (x - w[0]).T).T\\n\\n\\ndef _panel_b(panel, z):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v\'])\\n\\n    def loss(q):\\n        f = _design(np.cos(q * u))\\n        return np.mean((f @ _fit(f, v) - v)**2)\\n\\n    grid = np.linspace(.12, 7., 90)\\n    i = np.argmin([loss(q) for q in grid])\\n    q = minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),\\n                        method=\'bounded\', options={\'xatol\': 1e-8}).x\\n    c = np.cos(q * z)\\n    return _design(c) @ _fit(_design(np.cos(q * u)), v), c\\n\\n\\nfrom itertools import product\\nfrom scipy.linalg import solve\\nfrom scipy.optimize import minimize\\nfrom scipy.special import erf\\n\\n\\ndef _groups(z, b, c):\\n    blocks = []\\n    for x in (z, b, c):\\n        blocks.append(x)\\n        for q in (1., 1.5, 2., 2.5, 3., 4.):\\n            blocks.extend((np.tanh(q*x), np.sin(q*x), erf(q*x)))\\n        blocks.extend(x**p for p in (2, 3, 4, 5, 6))\\n        blocks.append(np.column_stack((np.sin(2*x), np.sin(4*x),\\n                                       np.cos(2*x), np.cos(4*x))))\\n    axes = {\'z\': z, \'b\': b, \'c\': c, \'t\': np.tanh(2*b), \'s\': np.sin(2.6*z)}\\n    for a, x in axes.items():\\n        for d, v in axes.items():\\n            if a <= d:\\n                blocks.append(np.column_stack([x[:, i]*v[:, j]\\n                              for i, j in product(range(x.shape[1]), range(v.shape[1]))\\n                              if a != d or i < j]))\\n        blocks.append(x)\\n        blocks.extend(x[:, j:j+1] for j in range(x.shape[1]))\\n    return [f for f in blocks if f.shape[1]]\\n\\n\\ndef _weights(blocks, n, targets):\\n    \\"\\"\\"Fit nonnegative group scales by regularized training-only leave-one-out.\\"\\"\\"\\n    a = np.asarray([f[:n] @ f[:n].T for f in blocks]).reshape(len(blocks), -1)\\n    eye = np.eye(n)\\n    y = targets / np.maximum(targets.std(axis=0), 1e-8)\\n    if y.shape[1] > 1:\\n        # Give the whole trajectory approximately the same weight as the scalar.\\n        y[:, 1:] *= targets[:, 1:].std(axis=0) / max(targets[:, 1:].std(), 1e-8)\\n        y[:, 1:] /= np.sqrt(y.shape[1]-1)\\n    penalty = .015\\n\\n    def objective(u):\\n        w = np.exp(u)\\n        gram = (w @ a).reshape(n, n) + 1. + .004*eye\\n        inverse = solve(gram, eye, assume_a=\'pos\', check_finite=False)\\n        alpha = inverse @ y\\n        diagonal = inverse.diagonal()\\n        residual = alpha / diagonal[:, None]\\n        da = 2*residual / diagonal[:, None]\\n        dq = -2*np.sum(residual**2, axis=1) / diagonal\\n        dg = -(inverse @ da) @ alpha.T - (inverse*dq) @ inverse\\n        return np.sum(residual**2) + penalty*np.sum(w), w*(a @ dg.ravel() + penalty)\\n\\n    result = minimize(objective, np.full(len(blocks), -4.), jac=True,\\n                      bounds=[(-12., 4.)]*len(blocks), method=\'L-BFGS-B\',\\n                      options={\'maxiter\': 170})\\n    return np.exp(result.x)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    tr, te = np.asarray(train[\'x\']), np.asarray(study[\'test\'][\'x\'])\\n    n = len(tr)\\n    z = _latent(study[\'panel_a\'], np.vstack((tr, te)))\\n    b, c = _panel_b(study[\'panel_b\'], z)\\n    blocks = [(f-f[:n].mean(axis=0)) / np.maximum(f[:n].std(axis=0), 1e-7)\\n              / np.sqrt(f.shape[1]) for f in _groups(z, b, c)]\\n    targets = np.column_stack((train[\'y\'], train[\'traj\']))\\n    weights = _weights(blocks, n, targets)\\n    # An offset makes the frozen quadratic kernel\'s linear term dominant:\\n    # (1 + offset + g/(2*(1+offset)))**2 = intercept + g + small remainder.\\n    dim = 1 + sum(f.shape[1] for f in blocks)\\n    offset = 16.\\n    features = np.column_stack([np.full(len(z), np.sqrt(dim*offset))] +\\n                 [f*np.sqrt(dim*w/(2*(1+offset))) for f, w in zip(blocks, weights)])\\n    return {\'train_features\': features[:n], \'test_features\': features[n:]}\\n", "node_01": "import numpy as np\\n\\n\\ndef _kernel(x, w):\\n    linear = x @ w.T / x.shape[1]\\n    r = .5 * np.abs(x[:, None, :] - w[None, :, :])\\n    g = (1 + r + r*r/3)*np.exp(-r)\\n    avg = g.mean(2)\\n    pair = avg*avg - (g*g).sum(2)/x.shape[1]**2\\n    return (1+linear)**2 + .30*avg + .30*pair\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    i, j = features[\'train_features\'], features[\'test_features\']\\n    k, ktest = _kernel(i, i), _kernel(j, i)\\n    return {\'k\': k, \'ktest\': ktest}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    tr = inputs[\'__input__\'][\'train\']\\n    kernels = inputs[\'node_01\']\\n    k, ktest = kernels[\'k\'], kernels[\'ktest\']\\n    n = k.shape[0]\\n    traj = np.asarray(tr[\'traj\'])\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    y = ktest @ np.linalg.solve(k + .004*np.eye(n), tr[\'y\'])\\n    state = (ktest @ np.linalg.solve(k + .004*np.eye(n), traj @ modes.T)) @ modes\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
