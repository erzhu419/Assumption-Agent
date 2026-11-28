import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares


def _cal_fit(sweeps, omit=None, degree=9, cross_degree=8, interactions=()):
    x = np.asarray(sweeps['ref_rosette'])
    y = np.asarray(sweeps['raw_rosette'])
    clean = np.max(np.abs(y), axis=1) < 8
    if omit is not None:
        clean &= ~omit
    x, y = x[clean], y[clean]
    isolated = np.count_nonzero(x, axis=1) == 1
    bank = sweeps['single_bank_sweeps']
    k, t, z = (np.asarray(bank[key]) for key in ('bank_id', 'ref_signal', 'raw_readout'))
    i, j = np.nonzero(x[isolated])
    k, t, z = np.r_[k, j], np.r_[t, x[isolated][i,j]], np.r_[z, y[isolated][i,j]]
    good = np.abs(z) < 8
    k, t, z = k[good], t[good], z[good]
    opp = (np.arange(4)+2)%4
    bias = np.array([np.median(y[isolated & (x[:,a]==0) & (x[:,opp[a]]==0),a]) for a in range(4)])
    powers = np.arange(1, degree+1, 2)
    V = t[:,None]**powers
    def residual(p):
        return bias[k] + p[k]*(V@np.r_[1.,p[4:]]) - z
    p = least_squares(residual, np.r_[np.ones(4), np.zeros(len(powers)-1)], gtol=1e-13).x
    def own(v):
        return np.polynomial.polynomial.polyval(v, np.ravel(np.column_stack((np.zeros(len(powers)),np.r_[1.,p[4:]]))))*p[:4]
    def design(x, a):
        v = x[:, (np.arange(4)+a)%4]
        w = v[:,2]
        return np.column_stack([w**i for i in range(1,cross_degree+1)] + [np.prod(v[:,q],axis=1) for q in interactions])
    r = y - bias - own(x)
    coeffs = [np.linalg.lstsq(design(x,a), r[:,a], rcond=None)[0] for a in range(4)]
    def predict(v):
        v = np.atleast_2d(v)
        return bias + own(v) + np.column_stack([design(v,a)@coeffs[a] for a in range(4)])
    return predict, bias, p, coeffs


def _invert(predict, observed):
    observed = observed.reshape(-1,4)
    out = []
    for row in observed:
        res = least_squares(lambda v: predict(v)[0]-row, np.zeros(4), gtol=1e-13, ftol=1e-13, xtol=1e-13)
        out.append(res.x)
    return np.asarray(out).ravel()
