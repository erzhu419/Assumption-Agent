import numpy as np
from scipy.optimize import least_squares

def _ls(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-10)[0]

def _aff(x):
    return np.column_stack((np.ones(len(x)),x))

def _calibrate(panel, x):
    u,v=panel['u'],panel['v']; pair=[2,3,0,1]
    y=v-.18*np.tanh(1.8*u)+.32*np.sin(2.4*u[:,pair])
    mask=np.ones(len(u),bool)
    for _ in range(4):
        off,gain=np.array([_ls(_aff(u[mask,j:j+1]),y[mask,j]) for j in range(4)]).T
        mask=np.max(abs(y-off-u*gain),axis=1)<.5
    z=x.copy()
    for _ in range(40):
        h=np.tanh(1.8*z)
        e=off+gain*z+.18*h-.32*np.sin(2.4*z[:,pair])-x
        j=np.zeros((len(x),4,4)); ix=np.arange(4)
        j[:,ix,ix]=gain+.324*(1-h*h)
        j[:,ix,pair]=-.768*np.cos(2.4*z[:,pair])
        z-=np.clip(np.linalg.solve(j,e[...,None])[...,0],-.28,.28)
    return z

def _outer(x,y):
    return (x[:,:,None]*y[:,None,:]).reshape(len(x),x.shape[1]*y.shape[1])

def _build(study,x):
    z=_calibrate(study['panel_a'],x)
    p=study['panel_b']; warp=lambda u:u-.39752*np.sin(2.8478*u-.002)
    w=_ls(_aff(warp(p['u'])),p['v']); b=_aff(warp(z))@w
    i,j=np.triu_indices(4)
    return z,b,np.column_stack((_aff(b),b[:,i]*b[:,j],_outer(b,z)))

def _scalar(a, b, y):
    n=len(y)
    scale=np.maximum(b[:n].std(axis=0), 1e-8)
    b=b/scale
    ab=a[:n]
    rng=np.random.default_rng(1)
    starts=np.r_[np.zeros((1,4)),np.eye(4),rng.normal(0,.5,(8,4))]
    predictions=[]
    for link in (np.sin,np.tanh):
        def design(p, q, base):
            return np.column_stack((base,link(p[0]+q@p[1:])))
        def residual(p):
            d=design(p,b[:n],ab)
            return np.r_[d@_ls(d,y)-y,.008*p]
        options=[least_squares(residual,np.r_[.1,start],max_nfev=120) for start in starts]
        fit=min(options,key=lambda opt:opt.cost)
        d=design(fit.x,b,a)
        predictions.append(d[n:]@_ls(d[:n],y))
    return np.mean(predictions,axis=0)


def solve_batch(study):
    train=study['train']; n=len(train['x']); x=np.concatenate((train['x'],study['test']['x']))
    _,b,a=_build(study,x)
    y_hat=_scalar(a,b,train['y'])
    _,_,vh=np.linalg.svd(train['traj'],full_matrices=False); modes=vh[:3]
    state_pred=(a[n:]@_ls(a[:n],train['traj']@modes.T))@modes
    last=state_pred[:,-1]
    return {'y_hat':y_hat,'state_pred':state_pred,'diagnostics_vec':np.array([y_hat.mean(),y_hat.std(),last.mean(),last.std()])}
