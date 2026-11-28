"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_13", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "ae40ec8fa696ac82520d45db3f9872117cd63223ffdb1c721ab024ad2d6362cb", "node_01": "e3df1300eed46285824aed6e1fbd2a06feb8781be6ee4d84b25a68fe9af04a31", "node_02": "6b2aed9db8bed4ef4803a05cc4d3b369a6ae5906b5a63da862248cde58bac85e"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent coordinates from the reference panels and assemble DAG features.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _fit(a, y):\\n    return np.linalg.lstsq(a, y, rcond=1e-10)[0]\\n\\n\\ndef _affine(a):\\n    return np.column_stack((np.ones(len(a)), a))\\n\\n\\ndef _sensor_basis(u, j):\\n    a, b = u[:, j], u[:, (j + 1) % u.shape[1]]\\n    return np.column_stack((np.ones(len(a)), a, b, a*b, a**3))\\n\\n\\ndef _inverse(panel, x):\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n    coeff = []\\n    for j in range(u.shape[1]):\\n        a = _sensor_basis(u, j)\\n        y = v[:, j]\\n        c = least_squares(lambda c: a @ c - y, _fit(a, y), loss=\'soft_l1\', f_scale=.003).x\\n        r = np.abs(a @ c - y)\\n        good = r <= max(.003, 5*np.median(r))\\n        coeff.append(_fit(a[good], y[good]))\\n    coeff = np.asarray(coeff)\\n    z = np.linalg.solve(np.diag(coeff[:, 1]) + np.roll(np.diag(coeff[:, 2]), 1, axis=1), (x-coeff[:, 0]).T).T\\n    for _ in range(15):\\n        nxt = np.roll(z, -1, axis=1)\\n        f = coeff[:, 0] + coeff[:, 1]*z + coeff[:, 2]*nxt + coeff[:, 3]*z*nxt + coeff[:, 4]*z**3 - x\\n        jac = np.zeros((len(z), z.shape[1], z.shape[1]))\\n        ii = np.arange(z.shape[1])\\n        jac[:, ii, ii] = coeff[:, 1] + coeff[:, 3]*nxt + 3*coeff[:, 4]*z*z\\n        jac[:, ii, (ii+1) % z.shape[1]] = coeff[:, 2] + coeff[:, 3]*z\\n        step = np.linalg.solve(jac, f[..., None])[..., 0]\\n        z -= step\\n        if np.max(np.abs(step), initial=0.) < 1e-12:\\n            break\\n    return z\\n\\n\\ndef _branch(panel, z):\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n    def basis(a, h):\\n        return np.column_stack((_affine(a), np.cos(h*a)))\\n    def err(h):\\n        a = basis(u, h)\\n        return np.sum((a @ _fit(a, v) - v)**2)\\n    grid = np.linspace(.2, 8., 80)\\n    j = int(np.argmin([err(h) for h in grid]))\\n    h = minimize_scalar(err, bounds=(grid[max(0,j-1)], grid[min(len(grid)-1,j+1)]), method=\'bounded\', options={\'xatol\': 1e-10}).x\\n    return basis(z, h) @ _fit(basis(u, h), v)\\n\\n\\ndef _readout(b, y):\\n    \\"\\"\\"Profile an affine readout, symmetric interaction, and one sine index.\\"\\"\\"\\n    pair = b[:, 0]*b[:, 2] + b[:, 1]*b[:, 3]\\n    base = np.column_stack((_affine(b), pair))\\n    n = len(y)\\n    a, x = base[:n], b[:n]\\n    q, _ = np.linalg.qr(a, mode=\'reduced\')\\n    centered_y = y - q @ (q.T @ y)\\n\\n    def residual(h):\\n        v = np.sin(h[0] + x @ h[1:])\\n        v = v - q @ (q.T @ v)\\n        return centered_y - v * ((v @ centered_y) / max(v @ v, 1e-20))\\n\\n    direction = _fit(a, y)[1:-1]\\n    starts = [np.r_[phase, mult*direction] for phase in (0., .35, -.35)\\n              for mult in (.85, 1.55, 2.2, 2.9)]\\n    best = min((least_squares(residual, h, max_nfev=120) for h in starts),\\n               key=lambda opt: opt.cost)\\n    return np.column_stack((base, np.sin(best.x[0] + b @ best.x[1:])))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = (np.asarray(study[k][\'x\'], float) for k in (\'train\', \'test\'))\\n    n = len(train)\\n    z = _inverse(study[\'panel_a\'], np.concatenate((train, test)))\\n    b = _branch(study[\'panel_b\'], z)\\n    state = _affine(b)\\n    readout = _readout(b, np.asarray(study[\'train\'][\'y\'], float))\\n    return {\'z\': state[:n], \'zt\': state[n:], \'readout_train\': readout[:n], \'readout_test\': readout[n:]}\\n", "node_01": "\\"\\"\\"Estimate separate state and scalar readout coefficients.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    features = inputs[\'node_00\']\\n    state = features[\'z\']\\n    traj = np.asarray(train[\'traj\'], float)\\n    a = features[\'readout_train\']\\n    y = np.asarray(train[\'y\'], float)\\n    scale = np.maximum(np.sqrt(np.mean(a*a, axis=0)), 1e-10)\\n    u, s, vh = np.linalg.svd(a/scale, full_matrices=False)\\n    readout = (vh.T @ ((s/(s*s + .001))*(u.T @ y)))/scale\\n    return {\'state_coef\': np.linalg.lstsq(state, traj, rcond=1e-10)[0], \'readout_coef\': readout}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    params = inputs[\'node_01\']\\n    yh = features[\'readout_test\'] @ params[\'readout_coef\']\\n    sp = features[\'zt\'] @ params[\'state_coef\']\\n    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])\\n    return {\'y_hat\': yh, \'state_pred\': sp, \'diagnostics_vec\': diag}\\n"}'

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
