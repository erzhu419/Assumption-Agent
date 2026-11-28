import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def _a_basis(x, j):
    h, q = x[:, j], x[:, (j + 1) % 4]
    return np.column_stack((np.ones(len(x)), h, q, h*q, h**3))


def _inverse(panel, x):
    u, v = panel['u'], panel['v']
    a = _linear(u)
    w = np.ones(len(u))
    for _ in range(10):
        r = np.linalg.norm(v-a@_fit(a*w[:,None],v*w[:,None]),axis=1)
        w = 1/np.sqrt(1+(2*r/max(np.median(r),1e-10))**2)
    k = np.argsort(r)[:-3]
    c = np.array([_fit(_a_basis(u[k], j), v[k,j]) for j in range(4)])
    z = x.copy()
    j = np.arange(4)
    qj = (j+1)%4
    for _ in range(12):
        q = z[:,qj]
        f = c[:,0]+c[:,1]*z+c[:,2]*q+c[:,3]*z*q+c[:,4]*z**3
        jac = np.zeros((len(x),4,4))
        jac[:,j,j] = c[:,1]+c[:,3]*q+3*c[:,4]*z*z
        jac[:,j,qj] = c[:,2]+c[:,3]*z
        z -= np.linalg.solve(jac,(f-x)[...,None])[...,0]
    return z


def _c1(x):
    return np.column_stack((np.ones(len(x)),x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]))


def _c2(x):
    return np.column_stack((_linear(x),np.cos(1.28*x)))


def _b(x):
    return np.column_stack((_linear(x),np.sin(2.8*x)))


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    x = np.concatenate((s['train']['x'],s['test']['x']))
    z = _inverse(s['panel_a'],x)
    p = s['panel_c']
    c = _c1(z) @ _fit(_c1(p['u']),p['v1'])
    d = _c2(z) @ _fit(_c2(p['u']),p['v2'])
    p = s['panel_b']
    cb = _fit(_b(p['u']),p['v'])
    b = _b(z)@cb
    g = np.column_stack((_linear(x),x*x,x*np.roll(x,1,axis=1),c,d))
    return {'g':g,'fd':b*c,'cd':_b(z+c)@cb,'y':s['train']['y'],'traj':s['train']['traj']}
