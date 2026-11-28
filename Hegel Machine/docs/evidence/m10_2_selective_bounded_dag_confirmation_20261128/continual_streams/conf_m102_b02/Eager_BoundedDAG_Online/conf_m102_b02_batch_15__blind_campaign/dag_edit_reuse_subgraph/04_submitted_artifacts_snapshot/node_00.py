"""Panel-calibrated coordinates for the frozen prediction nodes."""
import numpy as np
from scipy.optimize import minimize_scalar


def _design(*x):
    return np.column_stack((np.ones(len(x[0])), *x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-9)[0]


def _coordinates(s, x):
    p = s['panel_a']
    u, v = p['u'], p['v']
    w = []
    for j in range(u.shape[1]):
        f = _design(u[:, j], np.sin(2.4*u[:, j]), np.sin(2.4*u[:, (j+2)%4]))
        wc = _fit(f,v[:,j]); keep = abs(f@wc-v[:,j]) < 1.5
        w.append(_fit(f[keep],v[keep,j]))
    b,d,h,c = np.array(w).T
    z = (x-b)/d
    for _ in range(35):
        zp = np.roll(z,2,axis=1)
        r = b+d*z+h*np.sin(2.4*z)+c*np.sin(2.4*zp)-x
        a = d+2.4*h*np.cos(2.4*z)
        q = 2.4*c*np.cos(2.4*zp)
        ar,qr,rr = [np.roll(t,2,axis=1) for t in (a,q,r)]
        z -= np.clip((ar*r-q*rr)/(a*ar-q*qr),-.3,.3)
    p = s['panel_b']; f = lambda x: _design(np.cos(2.6*x))
    b = f(z)@_fit(f(p['u']),p['v'])
    p = s['panel_c']; f = lambda x: _design(x*np.roll(x,-1,axis=1))
    c = f(z)@_fit(f(p['u']),p['v1'])
    def r(x,a): return _design(x,x*x/(1+a*x*x))
    u,v = p['u'],p['v2']
    a = minimize_scalar(lambda t: np.linalg.norm(r(u,t)@_fit(r(u,t),v)-v),bounds=(.001,1),method='bounded').x
    d = r(z,a)@_fit(r(u,a),v)
    return z,b,c,d


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']; n = len(s['train']['x'])
    z,b,c,d = _coordinates(s,np.vstack((s['train']['x'],s['test']['x'])))
    f = np.c_[z,b,c,d]
    f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-6)
    return {'train_features':f[:n].copy(),'test_features':f[n:].copy()}
