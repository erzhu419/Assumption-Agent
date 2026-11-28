"""Calibrate frozen coordinates and predict scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf


def _ls(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-10)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)),x))


def _quad(x):
    return np.column_stack((_aff(x),*[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]))


def _warp(x,h,k):
    hx=h*x
    if k==0: return x*np.exp(-hx*hx)
    if k==1: return np.sin(hx)
    if k==2: return erf(hx)
    if k==3: return x*np.tanh(hx*hx)
    if k==4: return x/(1+hx**4)
    if k==5: return np.tanh(hx**3)
    if k==6: return np.sign(x)*np.log1p(abs(hx)**3)
    if k==7: return x/(1+hx*hx)
    if k==8: return np.arctan(hx)
    if k==9: return x/(1+abs(hx)**3)
    if k==10: return x/(1+hx*hx)**2
    return np.tanh(hx)


def _map(u,v,x,ks):
    scores=[]
    for k in ks:
        def design(a,h): return np.column_stack((_aff(a),_warp(a,h,k)))
        def loss(h):
            a=design(u,h)
            e=a@_ls(a,v)-v
            q=np.linalg.qr(a,mode='reduced')[0]
            return np.linalg.norm(e/np.maximum(.1,1-(q*q).sum(1))[:,None])
        grid=np.geomspace(.25,5,24)
        j=np.argmin([loss(h) for h in grid]); m=minimize_scalar(loss,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded')
        d=design(u,m.x)
        scores.append((m.fun,design(x,m.x)@_ls(d,v)))
    return min(scores,key=lambda z:z[0])[1]


def _coords(study,encoded):
    b=study['panel_b'];u,v=np.asarray(b['u']),np.asarray(b['v']);a=_ls(_aff(u),v)
    joined=np.concatenate((encoded['z'],encoded['zt']))
    raw=np.linalg.solve(a[1:].T,(joined-a[0]).T).T
    p=study.get('panel_c'); L=[_map(u,v,raw,range(12)),raw]
    if p is not None:
        L.append(_map(np.asarray(p['u']),np.asarray(p['v1']),raw,range(12)))
    return L


def _mat(x,mode,p=None):
    a=_aff(x); q=_quad(x)
    if mode==0: return q
    if mode==1:
        return np.column_stack((a,q[:,5:],x**3))
    if mode==2:
        return np.column_stack((q,x**3,x**4,*[x[:,i]**2*x[:,j] for i in range(4) for j in range(4) if i!=j]))
    if mode==3:
        return q/(1+(a@p)**2)[:,None]
    if mode==4:
        return np.column_stack((a,q[:,5:]/(1+(a@p)**2)[:,None]))
    if mode==5:
        return np.column_stack((a,q[:,5:]/(1+(x*x)@p)[:,None]))
    if mode==6:
        return q/(1+(x*x)@p)[:,None]
    if mode==7:
        return np.column_stack((a,q[:,5:]/(1+(x*x)@p)[:,None], x**3/(1+(x*x)@p)[:,None]))
    if mode==8:
        return np.column_stack((q,np.tanh(a@p)))
    return np.column_stack((q,np.sin(a@p)))


def _fit_predict(x,y,b,mode,reg=0.):
    if mode<3:
        d=_mat(x,mode); g=_mat(b,mode)
        if reg:
            penalty=np.r_[np.zeros(5),np.full(d.shape[1]-5,reg)]
            return g@np.linalg.solve(d.T@d+np.diag(penalty),d.T@y)
        return g@_ls(d,y)
    def res(p):
        d=_mat(x,mode,p)
        return np.r_[(d@_ls(d,y)-y).ravel(),.008*p]
    if 5<=mode<=7:
        starts=[np.full(4,.1),np.full(4,.8)]
        bounds=(0.,16.)
    else:
        starts=np.r_[np.eye(5)*.8,np.random.default_rng(186).normal(0,.6,(5,5))]
        bounds=(-8.,8.)
    opt=min((least_squares(res,start,bounds=bounds,max_nfev=45) for start in starts),key=lambda o:o.cost)
    return _mat(b,mode,opt.x)@_ls(_mat(x,mode,opt.x),y)


def _fit_model(x,y,b,key):
    mode,reg=key
    if mode<10: return _fit_predict(x,y,b,mode,reg)
    a=_aff(x); ab=_aff(b)
    x0=x[:,None,:]-x[None,:,:]; b0=b[:,None,:]-x[None,:,:]
    def kernel(d):
        r=mode/10*np.linalg.norm(d,axis=-1)
        return (1+r+r*r/3)*np.exp(-r)
    G=kernel(x0)+reg*np.eye(len(x)); K=kernel(b0)
    v=np.linalg.solve(G,y); A=np.linalg.solve(G,a); beta=_ls(a.T@A,a.T@v)
    return ab@beta+K@(v-A@beta)


def _predict(coords,y):
    y=np.asarray(y);n=len(y); idx=np.arange(n)
    folds=[(np.setdiff1d(idx,p),p) for p in np.array_split(np.random.default_rng(228).permutation(n),4)]
    keys=[(0,0.),(0,.06),(1,.01),(1,.1),(2,.02),(2,.18),(10,.001),(17,.01),(23,.025)]+[(i,0.) for i in range(3,10)]
    out=[]
    for i,xy in enumerate(coords):
        for key in keys:
            pred=np.empty_like(y)
            for a,b in folds:
                pred[b]=_fit_model(xy[a],y[a],xy[b],key)
            delta=np.abs(pred-y)
            score=delta.mean()+.15*np.quantile(delta,.9)
            out.append((score,i,key))
    best=sorted(out)[:3]; scores=np.array([item[0] for item in best]); w=np.exp(-12*(scores/scores[0]-1))
    return sum(wt*_fit_model(coords[i][:n],y,coords[i][n:],key) for wt,(_,i,key) in zip(w,best))/w.sum()


def run_node(inputs: dict) -> dict:
    study, encoded=inputs['__input__'],inputs['node_00']
    coords=_coords(study,encoded)
    train=study['train']; y=_predict(coords,train['y']); tr=_predict(coords,train['traj'])
    diag=np.array([y.mean(),y.std(),tr[:,-1].mean(),tr[:,-1].std()])
    return {'y_hat':y,'state_pred':tr,'diagnostics_vec':diag}
