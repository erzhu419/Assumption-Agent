import numpy as np


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    design = np.column_stack((np.ones(len(u)), u))
    weights = np.ones(len(u))
    for _ in range(12):
        coef = np.linalg.lstsq(design * weights[:, None], v * weights[:, None], rcond=None)[0]
        residual = np.linalg.norm(v - design @ coef, axis=1)
        weights = 1 / np.sqrt(1 + (residual / max(np.median(residual), .001)) ** 2)
    return coef[0], coef[1:]


def _assay(panel, key, terms):
    def design(u):
        return np.column_stack((np.ones(len(u)), u, *terms(u)))
    coef = np.linalg.lstsq(design(panel['u']), panel[key], rcond=None)[0]
    return lambda x: design(x) @ coef


def _blocks(study, mode):
    x = np.vstack((study['train']['x'], study['test']['x']))
    offset, gain = _calibrate(study['panel_a'])
    z = np.linalg.solve(gain.T, (x - offset).T).T
    b, c = study['panel_b'], study['panel_c']
    terms = (
        lambda u: (np.sin(2.815*u),),
        lambda u: (np.tanh(3.51*u), u/(1+(2.434*u)**2)),
        lambda u: (np.sin(2.815*u), u/(1+(2.434*u)**2)),
    )
    f = _assay(b, 'v', terms[mode])
    p = _assay(c, 'v1', lambda u: (np.sin(1.5*u), u*abs(u)))
    q = _assay(c, 'v2', lambda u: (u*u, np.cos(2.5*u)))
    B = f(z)
    C, D = p(z), q(z)
    H, J = p(B), q(B)
    return (z, B, C, D, H, J, p((z+B)/2), q((z+B)/2), q(H), f(B))


def _kernel(blocks, ids, kind, w, ratio):
    q = np.column_stack([blocks[i] for i in ids])
    q = (q - q.mean(axis=0)) / np.maximum(q.std(axis=0), 1e-8)
    delta = (q[:, None, :] - q[None, :, :])**2
    if kind == 'poly':
        return (1 + w * (q @ q.T) / q.shape[1]) ** 3
    if kind == 'matern':
        r = np.sqrt(w * delta.mean(axis=2))
        return (1 + r + r*r/3) * np.exp(-r)
    joint = np.exp(-w * delta.mean(axis=2))
    if kind == 'rbf':
        return joint
    return ratio * joint + (1-ratio) * np.exp(-w * delta).mean(axis=2)


CONFIG = (1, (0, 1, 2, 3, 4), 'poly', 0.12, 3.0, 0.7, (0, 1, 2, 3, 4), 1.0)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    mode, ids, kind, w, amplitude, ratio, linear, scale = CONFIG
    blocks = _blocks(study, mode)
    gram = _kernel(blocks, ids, kind, w, ratio)
    eigenvalues, vectors = np.linalg.eigh(gram)
    features = vectors * np.sqrt(amplitude * np.maximum(eigenvalues, 0))
    g = np.column_stack((np.ones(len(gram)), *[scale * blocks[i] for i in linear], features))
    return {'g': g, 'fd': blocks[1], 'cd': np.column_stack((blocks[2], blocks[3])),
            'y': study['train']['y'], 'traj': study['train']['traj']}
