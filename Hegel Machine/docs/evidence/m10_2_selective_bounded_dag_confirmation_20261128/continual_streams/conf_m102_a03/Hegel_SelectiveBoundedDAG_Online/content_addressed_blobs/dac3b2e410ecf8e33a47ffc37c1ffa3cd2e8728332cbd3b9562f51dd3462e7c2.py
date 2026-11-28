"""Reference-panel calibration and spatial feature blocks for the frozen fitter."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[key], float) for key in ('u', 'v'))
    good = np.linalg.norm(v, axis=1) < 5
    u, v = u[good], v[good]
    def design(z, j, h):
        a, b = z[:, j], z[:, (j+2)%4]
        return np.c_[np.ones(len(z)), a, b, a**3/(1+h*a*a), b**3/(1+h*b*b)]
    def profile(h, coeff=False):
        d = [design(u,j,h[0]) for j in range(4)]
        c = [_fit(a,v[:,j]) for j,a in enumerate(d)]
        return c if coeff else np.column_stack([a@q for a,q in zip(d,c)]).ravel()-v.ravel()
    h = least_squares(profile, [.48], bounds=(.02,3.)).x
    c = profile(h, True)
    def forward(z):
        return np.array([design(z[None],j,h[0])[0]@c[j] for j in range(4)])
    return np.array([least_squares(lambda z: forward(z)-row, row, gtol=1e-11).x for row in x])


def _bdesign(u, h, kind):
    a,b=h
    if kind=='sin': return np.c_[np.ones(len(u)),u,np.sin(a*u)]
    if kind=='sintanh': return np.c_[np.ones(len(u)),u,np.sin(a*u),np.tanh(b*u)]
    if kind=='sinatan': return np.c_[np.ones(len(u)),u,np.sin(a*u),np.arctan(b*u)]
    if kind=='sincos': return np.c_[np.ones(len(u)),u,np.sin(a*u),np.cos(b*u)-1]
    if kind=='warp': return np.c_[np.ones(len(u)),u+b*np.sin(a*u)]
    if kind=='rational': return np.c_[np.ones(len(u)),u+b*np.arctan(a*u)]
    return np.c_[np.ones(len(u)),u]


def _branch(panel,z,kind='sin',report=False):
    u,v=(np.asarray(panel[key],float) for key in ('u','v'))
    def residual(h):
        a=_bdesign(u,h,kind); return (a@_fit(a,v)-v).ravel()
    bounds=([.3,-3],[9,3]) if kind in ('warp','rational') else ([.3,.3],[9,9])
    o=min((least_squares(residual,[a,b],bounds=bounds,max_nfev=100) for a in (1.,2.9,5.5,8.) for b in ((-.4,.4) if kind in ('warp','rational') else (1.,3.))),key=lambda o:o.cost)
    h=o.x
    if report: print('B',kind,h,np.sqrt(np.mean(o.fun**2)))
    return _bdesign(z,h,kind)@_fit(_bdesign(u,h,kind),v), h


def _quad(x):
    i,j=np.triu_indices(x.shape[1]);return x[:,i]*x[:,j]


def _blocks(z,b,h):
    o=np.ones(len(z))
    R=np.c_[z,_quad(z),o,_quad(b),_quad(np.c_[z,b])[:,10:26],b]
    S=np.c_[z,o,_quad(z),_quad(b),b]
    return R,S



def _readout_dict(s,z,b,h):
    f=h[0]; H=np.sin(f*z); W=z-.4*H
    out={'z':z,'b':b,'h':H,'w':W,'wa':z-.30*H,'wb':z-.50*H,
         't':np.tanh(1.3*z),'q':z/(1+1.3*z*z),'e':np.cos(f*z)-1}
    for mode,name in [('rational','br'),('warp','bw'),('sintanh','bt'),('sinatan','ba')]:
        if name in ('z', 'ba', 'ba'):
            out[name]=_branch(s['panel_b'],z,mode)[0]
    return out


def _readout_form(d, spec):
    a,f,g,mode=spec; A,F,G=d[a],d[f],d[g]; n=len(A); one=np.ones(n)
    extra=[one]
    if mode in ('quadratic','mixed','cube','rational','harmonic','extended'):
        extra.append(_quad(G))
    if mode in ('mixed','cube','extended'):
        extra.append((A[:,:,None]*G[:,None,:]).reshape(n,-1))
    if mode in ('add','cube','extended'):
        extra.extend((A**3,G**3))
    if mode=='rational':
        extra.extend((A/(1+A*A),G/(1+G*G),np.tanh(G)))
    if mode=='harmonic':
        extra.extend((np.sin(2.85*A),np.cos(2.85*A)-1,np.sin(2.85*G),np.cos(2.85*G)-1))
    if mode=='extended':
        extra.extend((A**2*G,A*G**2))
    return np.c_[A,_quad(F),*extra,G]
def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; x=s['train']['x']; n=len(x)
    z=_calibrate(s['panel_a'],np.r_[x,s['test']['x']]); b,h=_branch(s['panel_b'],z)
    _,t=_blocks(z,b,h)
    r=_readout_form(_readout_dict(s,z,b,h), ('z', 'ba', 'ba', 'harmonic'))
    return {'readout_train':r[:n],'readout_test':r[n:],'z':t[:n],'zt':t[n:]}
