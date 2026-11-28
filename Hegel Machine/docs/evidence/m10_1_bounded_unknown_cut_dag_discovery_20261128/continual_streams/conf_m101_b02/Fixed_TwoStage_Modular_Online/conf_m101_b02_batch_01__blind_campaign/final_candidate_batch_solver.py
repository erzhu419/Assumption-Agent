"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "cca9764ed57126e34ba68139601d990dabdec48d3785f9f0fca5698f36f4d8f2", "node_01": "1ab0e98631990d7b38e6759d6dfa5f256b227ad1d5262b63fa625ace5de4742e"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _calibrate(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    design = np.column_stack((np.ones(len(u)), u))\\n    weights = np.ones(len(u))\\n    for _ in range(12):\\n        coef = np.linalg.lstsq(design * weights[:, None], v * weights[:, None], rcond=None)[0]\\n        residual = np.linalg.norm(v - design @ coef, axis=1)\\n        weights = 1 / np.sqrt(1 + (residual / max(np.median(residual), .001)) ** 2)\\n    return coef[0], coef[1:]\\n\\n\\ndef _assay(panel, key, terms):\\n    def design(u):\\n        return np.column_stack((np.ones(len(u)), u, *terms(u)))\\n    coef = np.linalg.lstsq(design(panel[\'u\']), panel[key], rcond=None)[0]\\n    return lambda x: design(x) @ coef\\n\\n\\ndef _blocks(study, mode):\\n    x = np.vstack((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    offset, gain = _calibrate(study[\'panel_a\'])\\n    z = np.linalg.solve(gain.T, (x - offset).T).T\\n    b, c = study[\'panel_b\'], study[\'panel_c\']\\n    terms = (\\n        lambda u: (np.sin(2.815*u),),\\n        lambda u: (np.tanh(3.51*u), u/(1+(2.434*u)**2)),\\n        lambda u: (np.sin(2.815*u), u/(1+(2.434*u)**2)),\\n    )\\n    f = _assay(b, \'v\', terms[mode])\\n    p = _assay(c, \'v1\', lambda u: (np.sin(1.5*u), u*abs(u)))\\n    q = _assay(c, \'v2\', lambda u: (u*u, np.cos(2.5*u)))\\n    B = f(z)\\n    C, D = p(z), q(z)\\n    H, J = p(B), q(B)\\n    return (z, B, C, D, H, J, p((z+B)/2), q((z+B)/2), q(H), f(B))\\n\\n\\ndef _kernel(blocks, ids, kind, w, ratio):\\n    q = np.column_stack([blocks[i] for i in ids])\\n    q = (q - q.mean(axis=0)) / np.maximum(q.std(axis=0), 1e-8)\\n    delta = (q[:, None, :] - q[None, :, :])**2\\n    if kind == \'poly\':\\n        return (1 + w * (q @ q.T) / q.shape[1]) ** 3\\n    if kind == \'matern\':\\n        r = np.sqrt(w * delta.mean(axis=2))\\n        return (1 + r + r*r/3) * np.exp(-r)\\n    joint = np.exp(-w * delta.mean(axis=2))\\n    if kind == \'rbf\':\\n        return joint\\n    return ratio * joint + (1-ratio) * np.exp(-w * delta).mean(axis=2)\\n\\n\\nCONFIG = (1, (0, 1, 2, 3, 4), \'poly\', 0.12, 3.0, 0.7, (0, 1, 2, 3, 4), 1.0)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    mode, ids, kind, w, amplitude, ratio, linear, scale = CONFIG\\n    blocks = _blocks(study, mode)\\n    gram = _kernel(blocks, ids, kind, w, ratio)\\n    eigenvalues, vectors = np.linalg.eigh(gram)\\n    features = vectors * np.sqrt(amplitude * np.maximum(eigenvalues, 0))\\n    g = np.column_stack((np.ones(len(gram)), *[scale * blocks[i] for i in linear], features))\\n    return {\'g\': g, \'fd\': blocks[1], \'cd\': np.column_stack((blocks[2], blocks[3])),\\n            \'y\': study[\'train\'][\'y\'], \'traj\': study[\'train\'][\'traj\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _ridge(g, target, penalty):\\n    n = len(target)\\n    training = g[:n]\\n    normal = training.T @ training + penalty * np.eye(g.shape[1])\\n    normal[0, 0] -= penalty\\n    return g[n:] @ np.linalg.solve(normal, training.T @ target)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    g, fd, cd = features[\'g\'], features[\'fd\'], features[\'cd\']\\n    traj = features[\'traj\']\\n    basis = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    target = np.column_stack((features[\'y\'], traj @ basis.T))\\n    base = _ridge(g, target, .0003)\\n    prediction = .5 * base + (\\n        _ridge(np.column_stack((g, fd)), target, .003)\\n        + _ridge(np.column_stack((g, cd)), target, .0003)\\n        + _ridge(np.column_stack((g, fd, cd)), target, .001)\\n    ) / 6\\n    y_hat = prediction[:, 0]\\n    state_pred = prediction[:, 1:] @ basis\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(),\\n                                         terminal.mean(), terminal.std()])}\\n"}'

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
