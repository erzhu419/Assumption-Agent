import numpy as np
from scipy.optimize import least_squares


def fit_a(panel):
    u,v = panel['u'],panel['v']
    d = np.stack((np.ones_like(u),u,(u-1)*np.roll(u,-1,axis=1),u**3),axis=-1)
    w = np.ones(len(u))
    for _ in range(24):
        c = np.stack([np.linalg.lstsq(d[:,j]*w[:,None],v[:,j]*w,rcond=None)[0] for j in range(4)],axis=1)
        e = np.linalg.norm(np.einsum('njk,kj->nj',d,c)-v,axis=1)
        w = 1/np.sqrt(1+(e/max(.001,2*np.median(e)))**2)
    k = np.argsort(e)[:-3]
    return np.stack([np.linalg.lstsq(d[k,j],v[k,j],rcond=None)[0] for j in range(4)],axis=1)


def inv_a(x,c):
    b,d,q,g = c
    idx = np.arange(4)
    mat = np.diag(d)
    mat[idx,(idx+1)%4] = -q
    initial = np.linalg.solve(mat,(x-b).T).T
    def resid(t,row):
        return b+d*t+q*(t-1)*np.roll(t,-1)+g*t**3-row
    def jac(t,row):
        j = np.diag(d+q*np.roll(t,-1)+3*g*t*t)
        j[idx,(idx+1)%4] = q*(t-1)
        return j
    return np.array([least_squares(resid,t,args=(row,),jac=jac,gtol=1e-12).x for row,t in zip(x,initial)])


def affine(x):
    return np.column_stack((np.ones(len(x)),x))


def pairs(x,kind):
    if kind==1:
        return x*x
    if kind==2:
        return x*np.roll(x,-1,axis=1)
    return np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])


def predict(study,options,t=None):
    mode,quad,weight,correction = options
    tr,te = study['train'],study['test']
    n = len(tr['x'])
    if t is None:
        t = inv_a(np.concatenate((tr['x'],te['x'])),fit_a(study['panel_a']))
    u,v = (study['panel_b'][k] for k in ('u','v'))
    traj = tr['traj']
    centered = traj-traj.mean(0)
    state = centered@np.linalg.svd(centered,full_matrices=False)[2][0]
    state /= np.std(state)
    def basis(x,p):
        a,f = p[:2]
        if mode==1:
            a = a + p[2:6]
            f = f + p[6:10]
        elif mode==2:
            x = x + p[2:6]
        w = x+a*np.sin(f*x)
        if mode==3:
            w = w + p[2]*np.sin(2*f*x) + p[3]*x*x
        if mode==4:
            w = x + a*x/(1+f*x*x) + p[2]*x**3
        if mode==5:
            w = x+a*np.tanh(f*x)+p[2]*x**3
        h = affine(w)
        return np.column_stack((h,pairs(x,quad))) if quad else h
    init = {0:[-.4,2.8],1:[-.4,2.8]+[0]*8,2:[-.4,2.8]+[0]*4,3:[-.4,2.8,0,0],4:[-1,.8,0],5:[-.75,1.4,0]}[mode]
    lo = [-3,.25]+[-.55]*(len(init)-2)
    hi = [1,5]+[.55]*(len(init)-2)
    def cal(p):
        d = basis(u,p)
        return d,np.linalg.lstsq(d,v,rcond=None)[0]
    def objective(p):
        d,c = cal(p)
        e = ((d@c-v)/np.std(v)).ravel()
        if weight:
            zz = affine(basis(t[:n],p)@c)
            h = zz@np.linalg.lstsq(zz,state,rcond=None)[0]-state
            e = np.r_[e,weight*h]
        return np.r_[e,.035*p[2:]]
    p = least_squares(objective,init,bounds=(lo,hi),max_nfev=110,ftol=1e-7).x
    z = basis(t,p)@cal(p)[1]
    if correction:
        # Estimate a smooth missing state direction, with the calibrated map as the offset.
        d = affine(z)
        if correction==1:
            h = np.column_stack((t,np.cos(2.8*t),pairs(t,3)))
        elif correction==2:
            h = np.column_stack((t,np.sin(2.8*t),np.cos(2.8*t),pairs(t,3),pairs(z,3)))
        else:
            h = np.column_stack((t,t**3,np.sin(2.8*t),np.cos(2.8*t),pairs(t,3),pairs(z,3),pairs(t-.4*np.sin(2.8*t),3)))
        h -= d@np.linalg.lstsq(d[:n],h[:n],rcond=None)[0]
        h /= np.maximum(h[:n].std(0),.055)
        direction = np.linalg.lstsq(d[:n],state,rcond=None)[0]
        err = state-d[:n]@direction
        lam = {1:9.,2:16.,3:28.}[correction]
        c = np.linalg.solve(h[:n].T@h[:n]+lam*np.eye(h.shape[1]),h[:n].T@err)
        z += (h@c)[:,None]*direction[1:]/np.dot(direction[1:],direction[1:])
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    z = predict(study,OPTIONS)
    n = len(study['train']['x'])
    return {'z_tr':z[:n],'z_te':z[n:]}
