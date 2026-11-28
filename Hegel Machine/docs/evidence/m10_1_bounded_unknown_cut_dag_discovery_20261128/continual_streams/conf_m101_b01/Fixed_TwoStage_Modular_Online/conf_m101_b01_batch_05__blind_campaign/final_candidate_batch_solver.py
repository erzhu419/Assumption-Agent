"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "551db397b6345c2e04ea0582406de298fee704e441ff9561b495779189e55b28", "node_01": "94d60a62e98ff41e61cff1b6a86773c18a6a70d7e409d019f356bc7e643795ef"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _align(panel, measured):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    rows, dims = u.shape\\n    i = np.arange(dims)\\n    design = np.zeros((rows, dims, 2 + 2*dims))\\n    design[:, :, 0] = np.roll(u, -1, axis=1) * (u - 1)\\n    design[:, :, 1] = u**3\\n    design[:, i, 2+i] = 1\\n    design[:, i, 2+dims+i] = u\\n    flat = design.reshape(rows*dims, -1)\\n    w = np.ones(rows)\\n    for _ in range(20):\\n        sw = np.repeat(np.sqrt(w), dims)\\n        coef = _fit(flat*sw[:, None], v.ravel()*sw)\\n        error = np.linalg.norm(design @ coef - v, axis=1)\\n        scale = max(2*np.median(error), 1e-8)\\n        w = 1/(1 + (error/scale)**2)\\n    keep = np.argsort(error)[:-3]\\n    g, d, *rest = _fit(design[keep].reshape(len(keep)*dims, -1), v[keep].ravel())\\n    offset, slope = np.array(rest).reshape(2, dims)\\n    z = (measured-offset)/slope\\n    jac = np.zeros((len(z), dims, dims))\\n    for _ in range(24):\\n        nxt = np.roll(z, -1, axis=1)\\n        residual = offset + slope*z + g*nxt*(z-1) + d*z**3 - measured\\n        if np.max(np.abs(residual)) < 1e-12:\\n            break\\n        jac[:, i, i] = slope + g*nxt + 3*d*z*z\\n        jac[:, i, (i+1) % dims] = g*(z-1)\\n        z -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -.45, .45)\\n    return z\\n\\n\\ndef _b_features(u, freq):\\n    return np.column_stack((np.ones(len(u)), u, np.sin(u*freq), u*u))\\n\\n\\ndef _activate(panel, z):\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def loss(freq):\\n        design = _b_features(u, freq)\\n        error = design @ _fit(design, v) - v\\n        return np.sum(error*error)\\n\\n    grid = np.linspace(2.0, 3.6, 33)\\n    freq = np.full(u.shape[1], min(grid, key=loss))\\n    for step in (.08, .04, .02, .01, .005, .0025):\\n        for _ in range(3):\\n            for j in range(len(freq)):\\n                tests = np.repeat(freq[None, :], 3, axis=0)\\n                tests[:, j] += (0, -step, step)\\n                freq = min(tests, key=loss)\\n    return _b_features(z, freq) @ _fit(_b_features(u, freq), v)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    x = np.concatenate((train[\'x\'], test[\'x\']))\\n    latent = _activate(study[\'panel_b\'], _align(study[\'panel_a\'], x))\\n    panel = study[\'panel_c\']\\n    return {\'latent\': latent, \'panel_coordinates\': panel[\'u\'],\\n            \'panel_v1\': panel[\'v1\'], \'panel_v2\': panel[\'v2\'],\\n            \'train_y\': train[\'y\'], \'train_trajectory\': train[\'traj\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _features(z, products):\\n    return np.column_stack((np.ones(len(z)), z, products))\\n\\n\\ndef _neighbor_products(z):\\n    return z*np.roll(z, -1, axis=1)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    u, z = data[\'panel_coordinates\'], data[\'latent\']\\n    c1 = np.linalg.lstsq(_features(u, _neighbor_products(u)), data[\'panel_v1\'], rcond=None)[0]\\n    c2 = np.linalg.lstsq(_features(u, u*u), data[\'panel_v2\'], rcond=None)[0]\\n    design = _features(z, np.column_stack((\\n        _features(z, _neighbor_products(z)) @ c1,\\n        _features(z, z*z) @ c2,\\n    )))\\n    target = np.column_stack((data[\'train_y\'], data[\'train_trajectory\']))\\n    n = len(target)\\n    coef = np.linalg.lstsq(design[:n], target, rcond=None)[0]\\n    pred = design[n:] @ coef\\n    y, state = pred[:, 0], pred[:, 1:]\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
