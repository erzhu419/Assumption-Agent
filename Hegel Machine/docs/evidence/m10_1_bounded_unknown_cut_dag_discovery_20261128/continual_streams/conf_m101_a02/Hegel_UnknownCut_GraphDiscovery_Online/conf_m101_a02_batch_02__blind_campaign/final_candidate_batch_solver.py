"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "d1f03ee4a574a398c740556a996324a09b27b30739626bfd83e776808c99f179", "node_01": "301dd96f03a5082e3549d67f9c368cfa6bcb75389cb60ea47c62d29477a64ea2", "node_02": "af847f9a1b10026272dc4d4ae4996cbf6bab674dad08c619f8ebce31947673ce"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _calibration(panel):\\n    coefficients = []\\n    for u, v in zip(panel[\'u\'].T, panel[\'v\'].T):\\n        design = np.column_stack((np.ones(len(u)), u))\\n        keep = np.arange(len(u))\\n        for _ in range(3):\\n            c = _fit(design[keep], v[keep])\\n            residual = np.abs(v - design @ c)\\n            keep = np.argsort(residual)[:3 * len(u) // 4]\\n        good = residual <= 8 * np.median(residual)\\n        coefficients.append(_fit(design[good], v[good]))\\n    return np.asarray(coefficients).T\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    offset, gain = _calibration(study[\'panel_a\'])\\n    return {\\n        \'cal_tr\': (study[\'train\'][\'x\'] - offset) / gain,\\n        \'cal_te\': (study[\'test\'][\'x\'] - offset) / gain,\\n        \'panel_b\': study[\'panel_b\'],\\n    }\\n", "node_01": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    calibrated = inputs[\'node_00\']\\n    panel = calibrated[\'panel_b\']\\n    mixer = np.linalg.lstsq(_affine(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    return {\\n        \'z_tr\': _affine(calibrated[\'cal_tr\']) @ mixer,\\n        \'z_te\': _affine(calibrated[\'cal_te\']) @ mixer,\\n    }\\n", "node_02": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _label_basis(x, z, frequency):\\n    a, b, c, d = z.T\\n    return np.column_stack((_affine(z), a * c, b * d,\\n                            np.sin(frequency * (x[:, 2] - d))))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    cal = inputs[\'node_00\']\\n    mix = inputs[\'node_01\']\\n    x, xt = cal[\'cal_tr\'], cal[\'cal_te\']\\n    z, zt = mix[\'z_tr\'], mix[\'z_te\']\\n    y = train[\'y\']\\n\\n    # Profile out the linear coefficients when estimating the sine frequency.\\n    frequency, best_loss = 2.5, np.inf\\n    for step in (0.10, 0.01):\\n        center = frequency\\n        for k in center + step * np.arange(-10, 11):\\n            basis = _label_basis(x, z, k)\\n            coef = _fit(basis, y)\\n            residual = basis @ coef - y\\n            loss = residual @ residual\\n            if loss < best_loss:\\n                frequency, best_loss, label_coef = k, loss, coef\\n    y_hat = _label_basis(xt, zt, frequency) @ label_coef\\n\\n    state_coef = _fit(_affine(x), train[\'traj\'])\\n    state_pred = _affine(xt) @ state_coef\\n    terminal = state_pred[:, -1]\\n    diagnostics = np.array((y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()))\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
