"""Panel-calibrated cyclic readouts for the frozen DAG."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _latent(panel, x):
    u, v = (np.asarray(panel[key], float) for key in ('u', 'v'))
    def design(a, i, w):
        p, q = a[:, i], a[:, (i+2)%4]
        return np.column_stack((np.ones(len(a)), p, p**3, q, np.sin(w*q), p*q))
    mask = np.max(abs(v), axis=1) < 4
    if mask.sum() < 12:
        mask = np.ones(len(u), dtype=bool)
    def calibrate(w):
        return np.array([_fit(design(u[mask], i, w), v[mask, i]) for i in range(4)])
    def forward(a, c, w):
        q = np.roll(a, 2, axis=-1)
        return c[:,0] + c[:,1]*a + c[:,2]*a**3 + c[:,3]*q + c[:,4]*np.sin(w*q) + c[:,5]*a*q
    def loss(w):
        return np.mean((forward(u[mask], calibrate(w), w) - v[mask])**2)
    w = minimize_scalar(loss, bounds=(1.5, 3.2), method='bounded').x
    c = calibrate(w)
    def jac(a):
        q = np.roll(a, 2)
        j = np.diag(c[:,1]+3*c[:,2]*a*a+c[:,5]*q)
        i = np.arange(4)
        j[i,(i+2)%4] = c[:,3]+w*c[:,4]*np.cos(w*q)+c[:,5]*a
        return j
    return np.asarray([least_squares(lambda a: forward(a,c,w)-b, np.clip(b,-1.28,1.28), jac=jac, bounds=(-1.3,1.3), gtol=1e-11).x for b in x]).reshape(-1,4)


def _panel(panel, key, z, transform, bounds):
    u, v = (np.asarray(panel[k], float) for k in ('u', key))
    def design(a, w):
        return np.column_stack((np.ones(len(a)), a, transform(w*a)))
    def loss(w):
        a = design(u,w)
        return np.mean((a@_fit(a,v)-v)**2)
    w = minimize_scalar(loss, bounds=bounds, method='bounded').x
    return design(z,w) @ _fit(design(u,w),v)


def _features(study, x):
    z = _latent(study['panel_a'], x)
    g = _panel(study['panel_b'], 'v', z, erf, (.15,4.5))
    j = _panel(study['panel_c'], 'v1', z, np.arctan, (.15,4.5))
    v = z*np.roll(g,1,axis=1)
    q = g[:,:2]*g[:,2:]
    core = np.column_stack((np.ones(len(z)),z,g,v,q))
    return np.column_stack((core,j,np.tanh(1.25*z))), core.shape[1]


def run_node(inputs: dict) -> dict:
    study, raw = inputs['__input__'], inputs['node_00']
    train = study['train']
    a, b = np.asarray(raw['a']), np.asarray(raw['a_test'])
    n = len(a)
    design, m = _features(study,np.vstack((a[:,1:],b[:,1:])))
    trajectory = np.asarray(train['traj'], float)
    coefficients = np.zeros((design.shape[1], 1+trajectory.shape[1]))
    coefficients[:,0] = _fit(design[:n], np.asarray(train['y'], float))
    coefficients[:m,1:] = _fit(design[:n,:m], trajectory)
    return {'test_features': design[n:].copy(), 'coefficients': coefficients}
