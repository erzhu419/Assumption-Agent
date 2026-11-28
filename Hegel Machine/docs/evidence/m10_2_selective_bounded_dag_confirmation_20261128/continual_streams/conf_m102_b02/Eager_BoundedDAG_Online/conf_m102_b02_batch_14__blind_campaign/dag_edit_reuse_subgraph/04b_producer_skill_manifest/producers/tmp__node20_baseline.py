import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-9)[0]


def _panel_a(study, x, verbose=False):
    p = study['panel_a']
    u, v = p['u'], p['v']
    keep = np.linalg.norm(u-v, axis=1) < 3
    u,v = u[keep],v[keep]
    # Diagonal and opposite-coordinate couplings share frequencies across axes.
    def parts(z, q, j):
        h = z[:, (j+2)%4]
        return _design(np.column_stack((z[:,j], np.sin(q[0]*z[:,j]), h, np.sin(q[1]*h))))
    def err(q):
        return np.concatenate([F@_fit(F,v[:,j])-v[:,j] for j in range(4) for F in [parts(u,q,j)]])
    opt = least_squares(err,[2.45,2.45],bounds=([.5,.5],[4,4]))
    q=opt.x
    co=np.stack([_fit(parts(u,q,j),v[:,j]) for j in range(4)])
    def forward(z):
        return np.column_stack([parts(z,q,j)@co[j] for j in range(4)])
    z=x-co[:,0]
    for _ in range(35):
        r=forward(z)-x
        J=np.zeros((len(z),4,4))
        for j in range(4):
            k=(j+2)%4
            J[:,j,j]=co[j,1]+co[j,2]*q[0]*np.cos(q[0]*z[:,j])
            J[:,j,k]=co[j,3]+co[j,4]*q[1]*np.cos(q[1]*z[:,k])
        d=np.linalg.solve(J, r[:,:,None])[:,:,0]
        z-=np.clip(d,-.2,.2)
    if verbose: print('panel A',q, np.sqrt(np.mean(opt.fun**2)),np.max(abs(forward(z)-x)),co)
    return z


def _panel_b(study,z):
    p=study['panel_b']; u,v=p['u'],p['v']
    def residual(q):
        f=_design(np.cos(q*u)); return np.mean((f@_fit(f,v)-v)**2)
    grid=np.linspace(.4,7,23); i=int(np.argmin([residual(q) for q in grid]))
    q=minimize_scalar(residual,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded').x
    return _design(np.cos(q*z))@_fit(_design(np.cos(q*u)),v)


def _groups(z,b):
    g={}
    for name,x in [('z',z),('b',b),('p',b+.3*z),('m',b-.3*z)]:
        g[name]=x
        g[name+'2']=x*x
        g[name+'3']=x*x*x
        for q in (0.8,1.4,2.15,2.6,3.5):
            for label,f in [('s',np.sin),('c',np.cos),('t',np.tanh),('e',erf),('a',np.arctan),('r',lambda a:a/(1+a*a)),('f',lambda a:a/np.sqrt(1+a*a))]:
                g[name+label+str(q)]=f(q*x)
    g['zb']=z*b
    g['zB']=(z[:,:,None]*b[:,None,:]).reshape(len(z),16)
    for name,x in [('z',z),('b',b)]:
        g[name+'cross']=np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)])
    return g


def _features(study):
    tr=study['train']; n=len(tr['x']); x=np.concatenate((tr['x'],study['test']['x']))
    z=_panel_a(study,x); b=_panel_b(study,z)
    return _groups(z,b),n


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; groups,n=_features(s)
    f=np.column_stack([groups[k] for k in ('z','b','zt2.15','bt2.15','zs2.6','bc2.15','zb')]); f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),.05)
    return {'train_features':f[:n], 'test_features':f[n:]}
