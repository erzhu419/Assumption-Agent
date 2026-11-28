"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "7df7a998c7fabec66ad192a936192314c818ec06e83afd76c4a43f0731abc84b", "node_01": "604e8aaf2f4d49e64817f8f32ff027513f63ab9c64f4d35582f0fe694d1483de"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _design(*blocks):\\n    return np.column_stack((np.ones(len(blocks[0])), *blocks))\\n\\n\\ndef _fit(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _ring(u):\\n    return u * np.roll(u, -1, axis=1)\\n\\n\\ndef _calibration(u):\\n    return _design(u, _ring(u), u**3)\\n\\n\\ndef _decode(panel, observed):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    d = _calibration(u)\\n    w = np.ones(len(u))\\n    for _ in range(20):\\n        c = _fit(d * w[:, None], v * w[:, None])\\n        r = np.linalg.norm(v - d @ c, axis=1)\\n        w = 1 / np.hypot(1, r / max(2 * np.median(r), 1e-10))\\n    keep = np.argsort(r)[:-3]\\n    c = _fit(d[keep], v[keep])\\n    z = np.linalg.solve(c[1:5].T, (observed - c[0]).T).T\\n    i = np.arange(z.shape[1])\\n    j = (i + 1) % z.shape[1]\\n    for _ in range(16):\\n        jac = np.broadcast_to(c[1:5], (len(z), 4, 4)).copy()\\n        jac += 3 * z[:, :, None]**2 * c[9:13]\\n        jac[:, i, :] += z[:, j, None] * c[5:9]\\n        jac[:, j, :] += z[:, i, None] * c[5:9]\\n        step = np.linalg.solve(jac.transpose(0, 2, 1), (_calibration(z) @ c - observed)[..., None])[..., 0]\\n        z -= step\\n        if np.max(np.abs(step)) < 1e-11:\\n            break\\n    return z\\n\\n\\ndef _v2(panel, z):\\n    u, v = panel[\'u\'], panel[\'v2\']\\n\\n    def basis(x, shape):\\n        even = x*x / (np.sqrt(1 + shape*x*x) + 1)\\n        return _design(x, even)\\n\\n    def loss(shape):\\n        d = basis(u, shape)\\n        return np.sum((d @ _fit(d, v) - v)**2)\\n\\n    shape = minimize_scalar(loss, bounds=(0.01, 3.0), method=\'bounded\').x\\n    return basis(z, shape) @ _fit(basis(u, shape), v)\\n\\n\\ndef _features(study):\\n    train, test = study[\'train\'], study[\'test\']\\n    z = _decode(study[\'panel_a\'], np.concatenate((train[\'x\'], test[\'x\'])))\\n    b, c = study[\'panel_b\'], study[\'panel_c\']\\n    b_hat = _design(z) @ _fit(_design(b[\'u\']), b[\'v\'])\\n    c1_hat = _design(_ring(z)) @ _fit(_design(_ring(c[\'u\'])), c[\'v1\'])\\n    c2_hat = _v2(c, z)\\n    return b_hat, c1_hat, c2_hat\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    b, c1, c2 = _features(study)\\n    n_train = len(train[\'x\'])\\n    # Scale the interactions for the frozen ridge readout while retaining B\'s\\n    # affine information. Center on training rows only.\\n    h = np.column_stack((0.09 * b, 2.0 * c1, c2))\\n    h -= h[:n_train].mean(axis=0)\\n    return {\'h\': h,\\n            \'n_train\': np.asarray(n_train),\\n            \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_01": "import numpy as np\\nfrom itertools import combinations_with_replacement\\n\\n\\ndef _readout_basis(x):\\n    mixed = np.column_stack([x[:, i]*x[:, j] for i, j in\\n                             combinations_with_replacement(range(x.shape[1]), 2)])\\n    return np.column_stack((np.ones(len(x)), x, .4*mixed))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    f = _readout_basis(upstream[\'h\'])\\n    n = upstream[\'n_train\']\\n    a, xt = f[:n], f[n:]\\n    gram = a@a.T\\n    cross = xt@a.T\\n    eye = np.eye(n)\\n    yh = cross@np.linalg.solve(gram+.0003*eye, upstream[\'y\'])\\n    state = cross@np.linalg.solve(gram+.001*eye, upstream[\'traj\'])\\n    end = state[:, -1]\\n    return {\'y_hat\': yh, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}\\n"}'

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
