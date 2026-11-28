"""Fit a paired nonlinear calibration, then predict dual-branch responses."""
import numpy as np
from scipy.optimize import least_squares


def _ols(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _design(z):
    return np.column_stack((np.ones(len(z)), z))


def _soft(z):
    return z / (1.0 + np.abs(z))


def _pair_terms(a, b):
    return np.stack((np.ones_like(a), a, b, a*a, a*b, b*b,
                     a**3, a*a*b, a*b*b, b**3), axis=-1)


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    cols = np.arange(u.shape[1]); nxt = (cols + 1) % u.shape[1]
    d = _design(u)
    good = np.ones(len(u), dtype=bool)
    for _ in range(6):
        linear = _ols(d[good], v[good])
        err = np.linalg.norm(d @ linear - v, axis=1)
        med = np.median(err)
        new = err <= max(.015, med + 6*1.4826*np.median(abs(err-med)))
        if new.sum() < 12 or np.array_equal(new, good):
            break
        good = new
    terms = _pair_terms(u, u[:, nxt])
    coef = np.stack([_ols(terms[good, j], v[good, j]) for j in cols])

    def model(z):
        return np.sum(_pair_terms(z, z[..., nxt])*coef, axis=-1)

    def jac(z):
        a, b = z, z[..., nxt]
        c = coef.T
        j = np.zeros(z.shape[:-1] + (len(cols), len(cols)))
        j[..., cols, cols] = c[1]+2*c[3]*a+c[4]*b+3*c[6]*a*a+2*c[7]*a*b+c[8]*b*b
        j[..., cols, nxt] = c[2]+c[4]*a+2*c[5]*b+c[7]*a*a+2*c[8]*a*b+3*c[9]*b*b
        return j

    def inverse(x):
        x = np.asarray(x, dtype=float)
        z = (x - linear[0]) @ np.linalg.pinv(linear[1:])
        for _ in range(24):
            r = model(z)-x
            if not len(z) or np.max(np.abs(r)) < 1e-11:
                break
            step = (np.linalg.pinv(jac(z)) @ r[..., None])[..., 0]
            z -= np.clip(step, -.5, .5)
        for i in np.flatnonzero(np.max(np.abs(model(z)-x), axis=1) > 1e-7):
            z[i] = least_squares(lambda q: model(q)-x[i], z[i], jac=jac).x
        return z
    return inverse


fun={'soft':lambda z:z/(1+abs(z)),'tanh':np.tanh,'rat':lambda z:z/(1+z*z),'sin':np.sin,'rat_even':lambda z:1/(1+z*z),'abs':np.abs,'square':np.square,'signed_sq':lambda z:z*abs(z),'cube':lambda z:z**3}
def design(x,p,ori,kind,mode):
 a,b=p; y=np.roll(x,-ori,axis=1); f=fun[kind]; z=f(a*y+b)
 if mode==0:return np.c_[np.ones(len(x)),x,z,x*y]
 if mode==1:return np.c_[np.ones(len(x)),x,z,x*z]
 if mode==2:return np.c_[np.ones(len(x)),x,z,x*y,x*z]
 if mode==3:return np.c_[np.ones(len(x)),x,z,x*y/(1+a*y*y)]
 if mode==4:return np.c_[np.ones(len(x)),x,x*y+b*z]
 if mode==5:return np.c_[np.ones(len(x)),x,x*z+b*f(a*y)]
 if mode==6:return np.c_[np.ones(len(x)),x,z,x*y/(1+a*(x-y)**2)]
 if mode==7:return np.c_[np.ones(len(x)),x,z,x*y+b*x*z]

def _branch2(u, v):
    def design2(x, p):
        return design(x, p, -1, 'tanh', 0)
    def residual(p):
        d = design2(u, p)
        return (d @ _ols(d,v)-v).ravel()
    q = least_squares(residual, [1.2009629146530316, -0.00016379671887329373], bounds=([.025,-3],[7,3])).x
    coef = _ols(design2(u,q),v)
    return lambda x: design2(x,q) @ coef

def _branches(panel):
    u = np.asarray(panel['u'], dtype=float)

    def fit(v, transform, start):
        def design(z, scale):
            return _design(np.column_stack((z, transform(scale[0] * z))))

        def residual(scale):
            d = design(u, scale)
            return (d @ _ols(d, v) - v).ravel()

        scale = least_squares(residual, [start], bounds=(0.05, 4.0), gtol=1e-13).x
        coef = _ols(design(u, scale), v)
        return lambda z: design(z, scale) @ coef

    return (fit(panel['v1'], _soft, 0.84),
            _branch2(u, panel['v2']))


def _features(study, x):
    z = _calibration(study['panel_a'])(x)
    branch1, branch2 = _branches(study['panel_c'])
    c, d = branch1(z), branch2(z)
    outer = (c[:, :, None] * d[:, None, :]).reshape(len(x), 16)
    base = np.column_stack((c, d))
    return (_design(np.column_stack((base, np.sum(c*d, axis=1)))),
            _design(np.column_stack((base, outer))))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    n = len(train['x'])
    fy, ft = _features(study, np.vstack((train['x'], study['test']['x'])))
    return {'y_hat': fy[n:] @ _ols(fy[:n], train['y']),
            'unprojected': ft[n:] @ _ols(ft[:n], train['traj'])}
