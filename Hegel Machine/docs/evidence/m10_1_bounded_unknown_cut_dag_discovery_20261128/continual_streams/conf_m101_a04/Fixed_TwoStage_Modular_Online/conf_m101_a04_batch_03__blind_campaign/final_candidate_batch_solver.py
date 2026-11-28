"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "42a19d85228d71058da541bec018ca35c48fe7b685eff70a712fdb8575fcc190", "node_01": "604e8aaf2f4d49e64817f8f32ff027513f63ab9c64f4d35582f0fe694d1483de"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(u):\\n    return np.column_stack((np.ones(len(u)), u))\\n\\n\\ndef _cyclic(u):\\n    return np.stack((np.ones_like(u), u, np.roll(u, -1, axis=1) * (u - 1), u**3), axis=-1)\\n\\n\\ndef _calibrate(panel, observed):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    design = _cyclic(u)\\n    w = np.ones(len(u))\\n    for _ in range(25):\\n        q = np.sqrt(w)\\n        g = np.array([np.linalg.lstsq(design[:, j] * q[:, None], v[:, j] * q, rcond=None)[0]\\n                      for j in range(u.shape[1])])\\n        e = np.linalg.norm(np.einsum(\'njk,jk->nj\', design, g) - v, axis=1)\\n        w = 1 / (1 + (e / max(2 * np.median(e), 1e-8))**2)\\n\\n    offset, linear, coupling, cubic = g.T\\n    idx = np.arange(u.shape[1])\\n    nxt = np.roll(idx, -1)\\n\\n    def forward(z):\\n        return offset + linear * z + coupling * z[nxt] * (z - 1) + cubic * z**3\\n\\n    def jac(z):\\n        j = np.diag(linear + coupling * z[nxt] + 3 * cubic * z*z)\\n        j[idx, nxt] += coupling * (z - 1)\\n        return j\\n\\n    j0 = jac(np.zeros(u.shape[1]))\\n    initial = np.linalg.solve(j0, (observed - offset).T).T\\n    return np.array([least_squares(lambda z: forward(z) - row, z0, jac=jac, gtol=1e-12).x\\n                     for row, z0 in zip(observed, initial)])\\n\\n\\ndef _c1(u, a):\\n    return u + a[1] * np.arcsinh(a[0] * u) + a[2] * u**3\\n\\n\\ndef _c2(u, a):\\n    return u + a[1] * (np.cos(a[0] * u) - 1) + a[2] * u**2\\n\\n\\ndef _project(u, v, z, transform, initial):\\n    def fit(a):\\n        design = _affine(transform(u, a))\\n        weights = np.linalg.lstsq(design, v, rcond=None)[0]\\n        return design @ weights - v, weights\\n\\n    a = least_squares(lambda a: fit(a)[0].ravel(), initial, max_nfev=180).x\\n    return _affine(transform(z, a)) @ fit(a)[1]\\n\\n\\ndef _branches(study, observed):\\n    z = _calibrate(study[\'panel_a\'], observed)\\n    b, c = study[\'panel_b\'], study[\'panel_c\']\\n    b_fit = np.linalg.lstsq(_affine(b[\'u\']), b[\'v\'], rcond=None)[0]\\n    v1 = _project(c[\'u\'], c[\'v1\'], z, _c1, [4.3, 4.4, -.56])\\n    v2 = _project(c[\'u\'], c[\'v2\'], z, _c2, [2.5, -.06, .35])\\n    return z, _affine(z) @ b_fit, v1, v2\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    z, b, v1, v2 = _branches(study, np.concatenate((train[\'x\'], test[\'x\'])))\\n    return {\'h\': np.column_stack((b, v1, v2)), \'n_train\': np.array(len(train[\'x\'])),\\n            \'y\': np.asarray(train[\'y\']), \'traj\': np.asarray(train[\'traj\'])}\\n", "node_01": "import numpy as np\\nfrom itertools import combinations_with_replacement\\n\\n\\ndef _readout_basis(x):\\n    mixed = np.column_stack([x[:, i]*x[:, j] for i, j in\\n                             combinations_with_replacement(range(x.shape[1]), 2)])\\n    return np.column_stack((np.ones(len(x)), x, .4*mixed))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    f = _readout_basis(upstream[\'h\'])\\n    n = upstream[\'n_train\']\\n    a, xt = f[:n], f[n:]\\n    gram = a@a.T\\n    cross = xt@a.T\\n    eye = np.eye(n)\\n    yh = cross@np.linalg.solve(gram+.0003*eye, upstream[\'y\'])\\n    state = cross@np.linalg.solve(gram+.001*eye, upstream[\'traj\'])\\n    end = state[:, -1]\\n    return {\'y_hat\': yh, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}\\n"}'

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
