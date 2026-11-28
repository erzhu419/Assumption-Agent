"""Robust calibration and cross-validated smooth prediction of two observables."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    z = np.empty_like(x, dtype=float)
    for j, (u, v) in enumerate(zip(np.asarray(panel['u']).T, np.asarray(panel['v']).T)):
        i, k = np.triu_indices(len(u), 1)
        du = u[i] - u[k]
        ok = np.abs(du) > .08
        gain = np.median((v[i[ok]] - v[k[ok]]) / du[ok])
        offset = np.median(v - gain*u)
        residual = v - offset - gain*u
        mad = np.median(np.abs(residual))
        keep = np.abs(residual) <= max(6*mad, .002)
        if keep.sum() >= 3:
            offset, gain = np.linalg.lstsq(np.column_stack((np.ones(keep.sum()), u[keep])), v[keep], rcond=None)[0]
        z[:, j] = (x[:, j] - offset)/gain
    return z


def _predict(study, x, target):
    n = len(target)
    z = _calibrate(study['panel_a'], x)
    p = study['panel_b']
    b = np.linalg.lstsq(_affine(p['u']), p['v'], rcond=None)[0]
    coords = (x, z, _affine(x)@b, _affine(z)@b, _affine(_affine(z)@b)@b)
    base = 1 + x@x[:n].T
    eye = np.eye(n)
    # Keep the affine mean common; compare nonlinear covariance geometries
    # using the exact deleted-row residual for each ridge system.
    def kernels():
        for a in coords:
            for v, lengths in ((a,(.8,1.15,1.7,2.5,3.5)), (np.sin(.65*a),(.52,.75,1.15,1.7)), (np.tanh(.65*a),(.52,.75,1.15,1.7)), (a/(1+.22*a*a),(.75,1.15,1.7,2.5))):
                g = v@v[:n].T
                norm = np.sum(v*v,axis=1)
                d = np.maximum(norm[:,None]+norm[None,:n]-2*g,0)
                for length in lengths:
                    yield np.exp(-d/(2*length**2))
    scores, predictions = [], []
    for kernel in kernels():
        k = base + kernel
        for ridge in (.0001,.0003,.001,.003,.01):
            c = cho_factor(k[:n]+ridge*eye, check_finite=False)
            alpha = cho_solve(c,target,check_finite=False)
            precision = cho_solve(c,eye,check_finite=False).diagonal()
            loo = np.abs(alpha/precision[:,None])
            scores.append(.8*loo.mean(axis=0)+.2*np.sqrt(np.mean(loo**2,axis=0)))
            predictions.append(k[n:]@alpha)
    scores = np.asarray(scores)
    predictions = np.asarray(predictions)
    best = scores.min(axis=0)
    weights = np.exp(-(scores-best)/np.maximum(.10*best, 1e-9))
    weights /= weights.sum(axis=0)
    return np.einsum('imc,ic->mc',predictions,weights)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.vstack((train['x'], study['test']['x']))
    traj = np.asarray(train['traj'])
    _, _, vt = np.linalg.svd(traj, full_matrices=False)
    basis = vt[:2]
    target = np.column_stack((train['y'], traj@basis.T))
    pred = _predict(study,x,target)
    y, state = pred[:,0], pred[:,1:]@basis
    end = state[:,-1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(),y.std(),end.mean(),end.std()])}
