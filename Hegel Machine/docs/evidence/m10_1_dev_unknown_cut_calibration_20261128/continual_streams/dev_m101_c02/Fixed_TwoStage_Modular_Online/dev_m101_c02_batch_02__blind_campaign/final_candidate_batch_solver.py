"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "725552af96e2dc75aeb57f19f03414b0f063db51a0ef69656c74c31a9eafc58b", "node_01": "0076f4b95b610e633066d1689fa5b6ae78c658268fcae2e01004e9b6a43df2c0"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a = study[\'panel_a\']\\n    good = np.max(np.abs(a[\'v\'] - a[\'u\']), axis=1) < 1.0\\n    u, v = a[\'u\'][good], a[\'v\'][good]\\n    centered = u - u.mean(axis=0)\\n    gain = np.sum(centered * (v - v.mean(axis=0)), axis=0) / np.sum(centered**2, axis=0)\\n    offset = v.mean(axis=0) - gain * u.mean(axis=0)\\n\\n    b = study[\'panel_b\']\\n    design = np.column_stack((np.ones(len(b[\'u\'])), b[\'u\']))\\n    operator = np.linalg.lstsq(design, b[\'v\'], rcond=None)[0]\\n\\n    def coordinates(x):\\n        return (x - offset) / gain @ operator[1:] + operator[0]\\n\\n    return {\'z_tr\': coordinates(study[\'train\'][\'x\']),\\n            \'z_te\': coordinates(study[\'test\'][\'x\'])}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _scalar_basis(z):\\n    return np.column_stack((_affine(z), z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))\\n\\n\\ndef _rational_basis(z, p):\\n    q = z @ p[:-1] + p[-1]\\n    denominator = 1.0 + q * q\\n    return np.column_stack((q, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3])) / denominator[:, None]\\n\\n\\ndef _predict_scalar(z, y, z_test):\\n    basis = _scalar_basis(z)\\n    orthogonal, _ = np.linalg.qr(basis, mode=\'reduced\')\\n\\n    def remove_basis(v):\\n        return v - orthogonal @ (orthogonal.T @ v)\\n\\n    target = remove_basis(y)\\n    direction = np.linalg.lstsq(basis, y, rcond=None)[0][1:5]\\n    direction /= max(np.linalg.norm(direction), 1e-12)\\n\\n    # Profile out the linear coefficients while fitting the shared denominator.\\n    def residual(p):\\n        terms = remove_basis(_rational_basis(z, p))\\n        weights = np.linalg.lstsq(terms, target, rcond=None)[0]\\n        return terms @ weights - target\\n\\n    fit = min(\\n        (least_squares(residual, np.r_[scale * direction, bias],\\n                       bounds=(-5.0, 5.0), max_nfev=300)\\n         for scale in (0.6, 1.4, 2.5) for bias in (0.0, 0.25, -0.25)),\\n        key=lambda result: result.cost,\\n    )\\n    design = np.column_stack((basis, _rational_basis(z, fit.x)))\\n    coeff = np.linalg.lstsq(design, y, rcond=None)[0]\\n    return np.column_stack((_scalar_basis(z_test), _rational_basis(z_test, fit.x))) @ coeff\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    coordinates = inputs[\'node_00\']\\n    z, z_test = coordinates[\'z_tr\'], coordinates[\'z_te\']\\n    y_hat = _predict_scalar(z, train[\'y\'], z_test)\\n    state_coeff = np.linalg.lstsq(_affine(z), train[\'traj\'], rcond=None)[0]\\n    state_pred = _affine(z_test) @ state_coeff\\n    terminal = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
