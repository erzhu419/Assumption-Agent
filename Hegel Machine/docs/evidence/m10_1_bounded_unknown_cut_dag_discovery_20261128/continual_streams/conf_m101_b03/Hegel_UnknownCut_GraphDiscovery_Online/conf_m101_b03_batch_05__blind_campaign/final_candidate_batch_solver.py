"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "b63773ba06bcdb09bbb4621325264fb52dbe04fd1bc82773c9f88844690ed4ab"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Predict batch responses from calibrated latent coordinates and operator interactions.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=1e-10)[0]\\n\\n\\ndef _latent(panel, x):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    i = np.arange(u.shape[1])\\n    j = (i + 1) % len(i)\\n    design = np.stack((np.ones_like(u), u, u[:, j], u*u[:, j], u**3), axis=-1)\\n    keep = np.ones(len(u), dtype=bool)\\n    for _ in range(8):\\n        w = np.stack([_fit(design[keep, k], v[keep, k]) for k in i], axis=1)\\n        residual = np.linalg.norm(np.sum(design*w.T, axis=-1)-v, axis=1)\\n        updated = residual <= max(4*np.median(residual), 1e-10)\\n        if np.array_equal(keep, updated):\\n            break\\n        keep = updated\\n    shift, gain, cross, product, cubic = w\\n    z = (x-shift)/gain\\n    for _ in range(24):\\n        next_z = z[:, j]\\n        residual = shift + gain*z + cross*next_z + product*z*next_z + cubic*z**3 - x\\n        if np.max(np.abs(residual), initial=0) < 1e-12:\\n            break\\n        jacobian = np.zeros((len(z), len(i), len(i)))\\n        jacobian[:, i, i] = gain + product*next_z + 3*cubic*z*z\\n        jacobian[:, i, j] = cross + product*z\\n        z -= np.linalg.solve(jacobian, residual[..., None])[..., 0]\\n    return z\\n\\n\\ndef _operators(panel, z):\\n    u, v = panel[\'u\'], panel[\'v1\']\\n\\n    def odd(x, p):\\n        rate, scale, weight = p\\n        return _affine(np.tanh(rate*x) + weight*x/(1+(scale*x)**2))\\n\\n    def residual(p):\\n        d = odd(u, p)\\n        return (d@_fit(d, v)-v).ravel()\\n\\n    fits = (least_squares(residual, p, bounds=([.2,.2,-30],[16,12,30]), max_nfev=180)\\n            for p in ((6,4,-5),(2,1,.5),(5,2,-4),(8,7,-4)))\\n    p = min(fits, key=lambda r:r.cost).x\\n    c = odd(z, p) @ _fit(odd(u, p), v)\\n    d = _affine(np.cos(2.4*z)) @ _fit(_affine(np.cos(2.4*u)), panel[\'v2\'])\\n    return c, d\\n\\n\\ndef _outer(a, b):\\n    return (a[:, :, None]*b[:, None, :]).reshape(len(a), a.shape[1]*b.shape[1])\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    tr = study[\'train\']\\n    n = len(tr[\'x\'])\\n    x = np.concatenate((tr[\'x\'], study[\'test\'][\'x\']))\\n    z = _latent(study[\'panel_a\'], x)\\n    c, d = _operators(study[\'panel_c\'], z)\\n    cyclic = np.sum(z * np.roll(c, 1, axis=1), axis=1)\\n    base = np.column_stack((np.ones(len(z)), c, d, cyclic))\\n    scalar = np.column_stack((base, np.sum(c*d, axis=1)))\\n    state = np.column_stack((base, _outer(c, d)))\\n    y_hat = scalar[n:] @ _fit(scalar[:n], tr[\'y\'])\\n    state_pred = state[n:] @ _fit(state[:n], tr[\'traj\'])\\n    endpoint = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
    outputs = {'__input__': deepcopy(study)}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        if len(_SPEC_OBJ['topo_order']) == 1 and srcs == ['__input__']:
            node_in = deepcopy(outputs['__input__'])
        else:
            node_in = {s: deepcopy(outputs[s]) for s in srcs}
        in_dig = _canon_digest(node_in)
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Anti-identity-shell check: intermediate node cannot merely echo its raw predecessor dict
            echoed = (out_dig == in_dig) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            if echoed:
                raise RuntimeError(f'DAG node {nid} failed anti-identity-shell check')
            non_identity_checks[nid] = True
        outputs[nid] = out_v
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
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
