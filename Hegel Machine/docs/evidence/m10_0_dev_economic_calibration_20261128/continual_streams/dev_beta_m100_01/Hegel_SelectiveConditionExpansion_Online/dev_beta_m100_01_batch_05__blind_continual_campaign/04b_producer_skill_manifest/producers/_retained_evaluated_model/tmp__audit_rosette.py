import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _invert, _basis, _PAIR, _gross_good
from batch_solver import solve_batch

for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    s = item['study']
    c = s['sensor_calibration_sweeps']
    X, Y = c['ref_rosette'], c['raw_rosette']
    axis = np.count_nonzero(X, axis=1) == 1
    good = np.all(_gross_good(Y), axis=1)
    rng = np.random.default_rng(52)
    folds = [np.array_split(rng.permutation(np.flatnonzero(m & good)), 5) for m in (axis, ~axis)]
    mods = ('affine', 'harmonic', 'harmonic+rational', 'harmonic+pairwise')
    errors = {m: np.zeros_like(Y) for m in mods}
    inverse = np.zeros_like(X)
    def design(x, f, mod):
        z = x[:, _PAIR]
        d = _basis(x, z, f)
        if mod == 'harmonic': return d
        if mod == 'affine': return d[..., [0, 1, 3]]
        if mod == 'harmonic+rational':
            return np.concatenate((d, np.stack([a**i/(1+a*a) for a in (x,z,x-z,x+z) for i in (1,2)], axis=-1)), axis=-1)
        return np.concatenate((d, np.stack([np.sin(f[0]*(x-z)), np.sin(f[0]*(x+z)), np.cos(f[1]*(x-z))-1, np.cos(f[1]*(x+z))-1], axis=-1)), axis=-1)
    for a,b in zip(*folds):
        test = np.r_[a,b]; train = np.ones(len(X), bool); train[test] = False
        f,p = _fit(dict(c, ref_rosette=X[train], raw_rosette=Y[train]))
        errors['harmonic'][test] = np.einsum('nck,ck->nc', design(X[test],f,'harmonic'),p)-Y[test]
        inverse[test] = _invert(Y[test],f,p)-X[test]
        sb = c['single_bank_sweeps']; ids = sb['bank_id']; u = sb['ref_signal']; v = sb['raw_readout']
        for mod in ('affine', 'harmonic+rational', 'harmonic+pairwise'):
            tr = train & good
            d, dt = design(X[tr],f,mod), design(X[test],f,mod)
            for j in range(4):
                m = ids == j; m[m] = _gross_good(v[m]); bank_x = np.eye(4)[j] * u[m,None]
                d0 = np.vstack((d[:,j],design(bank_x,f,mod)[:,j])); y0 = np.r_[Y[tr,j], v[m]]
                errors[mod][test,j] = dt[:,j] @ np.linalg.lstsq(d0,y0,rcond=None)[0] - Y[test,j]
    for k,r in errors.items():
        print(k,'forward RMSE axis/mixed', [float(np.sqrt(np.mean(r[m & good]**2))) for m in (axis,~axis)])
    print('inverse RMSE axis/mixed', [float(np.sqrt(np.mean(inverse[m & good]**2))) for m in (axis,~axis)])
    print('field scope keys', list(s))
    f,p = _fit(c)
    diag_min = p[:,1]-abs(p[:,2]*f[0]); off_max = abs(p[:,3])+abs(p[:,4]*f[1])
    print('global row diagonal-dominance margins',diag_min-off_max)
    assert np.min(diag_min-off_max) > 0
    # Exercise the actual response at the lowest diagonal gain and highest
    # paired derivative, in addition to the observed reference extrema.
    u = np.linspace(min(X.min(),-np.pi/f[0]), max(X.max(),np.pi/f[0]), 51)
    a,b = np.meshgrid(np.r_[u,np.pi/f[0]],np.r_[u,-np.pi/(2*f[1])]); q=np.c_[a.ravel(), b.ravel(), b.ravel(), a.ravel()]
    qy = np.einsum('nck,ck->nc',_basis(q,q[:,_PAIR],f),p)
    e = np.max(np.abs(_invert(qy,f,p)-q)); print('stress-grid max inverse error',e); assert e < 1e-9
    print('real batch certificate gap', (lambda k: k['primal_objective']-k['dual_lower_bound'])(solve_batch(s)['kkt_certificate']))
