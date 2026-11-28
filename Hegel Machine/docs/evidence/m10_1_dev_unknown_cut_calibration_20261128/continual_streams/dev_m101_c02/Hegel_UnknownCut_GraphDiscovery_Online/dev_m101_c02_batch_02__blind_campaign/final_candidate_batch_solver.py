"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "7e673fba99d53c60db378eb8a4e97f6d82c561f690c883e5c312d6e9c26b5496", "node_01": "8e24ebf14ed0da778d1e99da53e9a706bc78a63ef54f225bfa17fff5c1748c39", "node_02": "5d5ee9c51f4c8e08b5e70011b436e21cfc0f5d60bc7c809cf66d92bd87f7230a"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a = study[\'panel_a\']\\n    good = np.max(np.abs(a[\'v\'] - a[\'u\']), axis=1) < 1.0\\n    u, v = a[\'u\'][good], a[\'v\'][good]\\n    centered = u - u.mean(axis=0)\\n    gain = np.sum(centered * (v - v.mean(axis=0)), axis=0) / np.sum(centered**2, axis=0)\\n    offset = v.mean(axis=0) - gain * u.mean(axis=0)\\n    return {\\n        \'u_tr\': (study[\'train\'][\'x\'] - offset) / gain,\\n        \'u_te\': (study[\'test\'][\'x\'] - offset) / gain,\\n        \'panel_b\': study[\'panel_b\'],\\n    }\\n", "node_01": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    calibrated = inputs[\'node_00\']\\n    b = calibrated[\'panel_b\']\\n    design = np.column_stack((np.ones(len(b[\'u\'])), b[\'u\']))\\n    operator = np.linalg.lstsq(design, b[\'v\'], rcond=None)[0]\\n    return {\\n        \'z_tr\': calibrated[\'u_tr\'] @ operator[1:] + operator[0],\\n        \'z_te\': calibrated[\'u_te\'] @ operator[1:] + operator[0],\\n    }\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _mixed(u, z, z_mean):\\n    products = (u[:, :, None] * (z - z_mean)[:, None, :]).reshape(len(u), -1)\\n    return np.column_stack((_affine(u), products))\\n\\n\\ndef _scalar(u_tr, z_tr, u_te, z_te, y):\\n    z_mean = z_tr.mean(axis=0)\\n    design = _mixed(u_tr, z_tr, z_mean)\\n    query = _mixed(u_te, z_te, z_mean)\\n    # The affine mapping makes some mixed products linearly dependent.\\n    _, singular, right = np.linalg.svd(design, full_matrices=False)\\n    basis = right[singular > 1e-10 * singular[0]].T\\n    design, query = design @ basis, query @ basis\\n    scale = max(2.0, 1.05 * np.max(np.abs(y)))\\n    initial = np.r_[scale, np.linalg.lstsq(design, y / scale, rcond=None)[0]]\\n\\n    def residual(params):\\n        return np.r_[params[0] * np.arctan(design @ params[1:]) - y,\\n                     0.003 * (params - initial)]\\n\\n    def jacobian(params):\\n        q = design @ params[1:]\\n        derivative = params[0] / (1.0 + q * q)\\n        return np.vstack((np.column_stack((np.arctan(q), derivative[:, None] * design)),\\n                          0.003 * np.eye(len(params))))\\n\\n    fitted = least_squares(residual, initial, jac=jacobian, max_nfev=160).x\\n    return fitted[0] * np.arctan(query @ fitted[1:])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    calibrated = inputs[\'node_00\']\\n    mapped = inputs[\'node_01\']\\n    u_tr, u_te = calibrated[\'u_tr\'], calibrated[\'u_te\']\\n    y_hat = _scalar(u_tr, mapped[\'z_tr\'], u_te, mapped[\'z_te\'], train[\'y\'])\\n    trajectory = np.linalg.lstsq(_affine(u_tr), train[\'traj\'], rcond=None)[0]\\n    state_pred = _affine(u_te) @ trajectory\\n    terminal = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
