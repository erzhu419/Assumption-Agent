"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_11", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "73de9cbb79a3791be3c4e418b8bc06eab2a88cb92af500a5e61575d8043bda7d"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Infer operators from the calibration panels, then fit the observable readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _bias(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _coordinates(study, x):\\n    p = study[\'panel_a\']\\n    u, v = np.asarray(p[\'u\']), np.asarray(p[\'v\'])\\n    def a_features(z, j):\\n        f, g = z[:, j], z[:, (j+1) % 4]\\n        return np.column_stack((np.ones(len(z)), f, g, f*g, f**3))\\n    designs = [a_features(u, j) for j in range(4)]\\n    keep = np.arange(len(u))\\n    for _ in range(7):\\n        pars = np.array([_ls(f[keep], v[keep, j]) for j, f in enumerate(designs)])\\n        residual = np.linalg.norm(np.column_stack([f @ pars[j] for j, f in enumerate(designs)])-v, axis=1)\\n        new_keep = np.flatnonzero(residual <= max(0.001, 5*np.median(residual)))\\n        if len(new_keep) < 12 or np.array_equal(new_keep, keep):\\n            break\\n        keep = new_keep\\n    # Invert the cyclic cubic map by Newton iterations, independently per row.\\n    z = x.copy()\\n    for _ in range(25):\\n        w = np.roll(z, -1, axis=1)\\n        res = pars[:,0] + pars[:,1]*z + pars[:,2]*w + pars[:,3]*z*w + pars[:,4]*z**3-x\\n        if np.max(np.abs(res), initial=0.) < 1e-12:\\n            break\\n        jac = np.zeros((len(z),4,4))\\n        j = np.arange(4)\\n        jac[:,j,j] = pars[:,1]+pars[:,3]*w+3*pars[:,4]*z*z\\n        jac[:,j,(j+1)%4] = pars[:,2]+pars[:,3]*z\\n        z -= np.clip(np.linalg.solve(jac, res[...,None])[...,0], -0.7, 0.7)\\n    return z\\n\\n\\ndef _panels(study, z):\\n    c = study[\'panel_c\']\\n    u, v1, v2 = (np.asarray(c[k]) for k in (\'u\', \'v1\', \'v2\'))\\n    # The sine component shares the mixing matrix with tanh after a cyclic shift.\\n    def first(u, p):\\n        a, b, c = p\\n        return _bias(np.tanh(a*u)+c*np.roll(np.sin(b*u), 1, axis=1))\\n    def second(u, p):\\n        return _bias(np.cos(p[0]*u))\\n    def proj(p, make, v):\\n        f = make(u, p)\\n        return (f @ _ls(f, v)-v).ravel()\\n    p1 = least_squares(proj, [1.75,2.1,0.3], args=(first,v1), bounds=([1,1.5,0],[2.5,2.8,.6])).x\\n    p2 = least_squares(proj, [2.4], args=(second,v2), bounds=([1.8],[3.])).x\\n    return np.column_stack([f(z,p) @ _ls(f(u,p),v) for f,p,v in ((first,p1,v1),(second,p2,v2))])\\n\\n\\ndef _readout_features(w):\\n    # The channels interact through their matched contraction and one\\n    # additional directed coupling in the trajectory readout.\\n    a, b = w[:, :4], w[:, 4:]\\n    return _bias(np.column_stack((w, np.einsum(\'ij,ij->i\', a, b), a[:, 0]*b[:, 1])))\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    tr = study[\'train\']\\n    x, xt = np.asarray(tr[\'x\']), np.asarray(study[\'test\'][\'x\'])\\n    w = _panels(study, _coordinates(study, np.vstack((x,xt))))\\n    f = _readout_features(w)\\n    fit = f[:len(x)]\\n    traj = np.asarray(tr[\'traj\'])\\n    _, _, temporal = np.linalg.svd(traj, full_matrices=False)\\n    basis = temporal[:2]\\n    y = f[len(x):, :-1] @ _ls(fit[:, :-1], np.asarray(tr[\'y\']))\\n    state = f[len(x):] @ _ls(fit, traj @ basis.T) @ basis\\n    return {\'y_hat\':y, \'state_pred\':state, \'diagnostics_vec\':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}\\n"}'

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
