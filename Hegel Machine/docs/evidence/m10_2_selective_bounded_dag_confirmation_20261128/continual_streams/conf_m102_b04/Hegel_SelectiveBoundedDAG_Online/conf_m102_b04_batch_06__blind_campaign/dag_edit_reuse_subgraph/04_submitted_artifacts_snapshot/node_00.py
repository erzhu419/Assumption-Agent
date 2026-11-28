"""Recover the shared latent coordinates from the reference calibration."""
import numpy as np
from scipy.optimize import least_squares


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel = study['panel_a']
    u, v = np.asarray(panel['u'], float), np.asarray(panel['v'], float)
    d = u.shape[1]
    nxt = (np.arange(d) + 1) % d

    def terms(q, k):
        return np.column_stack((np.ones(len(q)), q, q[:, k]**3, q[:, k]*q[:, nxt[k]]))

    # Gross reference corruptions affect whole rows. Refine the joint inlier
    # mask after fitting the sparse cyclic model to the uncorrupted majority.
    delta = np.linalg.norm(v-u, axis=1)
    keep = delta <= max(3., 4.*np.median(delta))
    coef = np.stack([np.linalg.lstsq(terms(u[keep], k), v[keep,k], rcond=None)[0] for k in range(d)], axis=1)

    def forward(q):
        return coef[0] + q @ coef[1:d+1] + coef[-2]*q**3 + coef[-1]*q*q[nxt]

    errors = np.linalg.norm(np.array([forward(q) for q in u])-v, axis=1)
    good = errors <= max(0.003, 6.*np.median(errors))
    if np.count_nonzero(good) >= d+3:
        coef = np.stack([np.linalg.lstsq(terms(u[good], k), v[good,k], rcond=None)[0] for k in range(d)], axis=1)

    def jac(q):
        out = coef[1:d+1].T.copy()
        k = np.arange(d)
        out[k,k] += 3.*coef[-2]*q*q + coef[-1]*q[nxt]
        out[k,nxt] += coef[-1]*q
        return out

    train = np.asarray(study['train']['x'], float)
    test = np.asarray(study['test']['x'], float)
    x = np.concatenate((train, test), axis=0)
    init = np.linalg.lstsq(coef[1:d+1].T, (x-coef[0]).T, rcond=None)[0].T
    latent = np.empty_like(x)
    for i, (target, q) in enumerate(zip(x, init)):
        latent[i] = least_squares(lambda z: forward(z)-target, q, jac=jac, max_nfev=50,
                                  ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return {'cubic_train': latent[:len(train)]**3, 'cubic_test': latent[len(train):]**3}
