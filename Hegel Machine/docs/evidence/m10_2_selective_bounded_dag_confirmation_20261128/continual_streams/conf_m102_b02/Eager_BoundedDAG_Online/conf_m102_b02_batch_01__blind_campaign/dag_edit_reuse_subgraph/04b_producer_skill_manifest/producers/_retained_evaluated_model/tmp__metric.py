import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve

def _bank(x, study):
    blocks=[x]
    for pk,vk,kind in [('panel_b','v',0),('panel_c','v1',1),('panel_c','v2',2)]:
        p=study[pk]; u=p['u']
        def design(a):
            h=(a**3/(1+0.8*a*a) if kind==0 else np.arcsinh(3.5*a) if kind==1 else a*a)
            return np.c_[np.ones(len(a)),a,h]
        c=np.linalg.lstsq(design(u),p[vk],rcond=None)[0]
        blocks.append(design(x)@c-c[0])
    blocks.extend([x**2,x**3/(1+x*x), np.arcsinh(3.5*x)])
    return blocks

def _fit(z, y, start, lam, nit):
    n=len(y); d=start.shape[1]; eye=np.eye(n)
    def fg(w):
        c=w.reshape(-1,d); v=z@c; h=np.arcsinh(3.5*v); hv=h*v; hh=h*h
        ll=v@v.T; hl=h@h.T; b=hl+0.25*ll
        g=1+ll+hl+hh@hh.T+hv@hv.T+0.2*b*b
        g.flat[::n+1]+=1e-6*np.diag(g).mean()
        gi=cho_solve(cho_factor(g,check_finite=False),eye,check_finite=False)
        alpha=gi@y; q=gi.diagonal(); e=alpha/q[:,None]
        loss=np.mean(e*e)+lam*np.sum((c-start)**2)
        da=2*e/q[:,None]/e.size
        S=da@y.T
        S.flat[::n+1]-=np.sum(da*e,axis=1)
        S=-gi@S@gi; S=S+S.T
        S.flat[::n+1]+=1e-6*np.trace(S)/n
        dh=3.5/np.sqrt(1+(3.5*v)**2)
        dv=(S*(1+0.1*b))@v+((S*(1+0.4*b))@h)*dh+(S@hh)*(2*h*dh)+(S@hv)*(h+v*dh)
        return loss,(z.T@dv+2*lam*(c-start)).ravel()
    o=minimize(fg,start.ravel(),jac=True,method='L-BFGS-B',options={'maxiter':nit,'ftol':1e-10}); return o.fun,o.x.reshape(start.shape)

def _map(x, study, mode=0):
    n=len(study['train']['x']); tr=study['train']; a=np.asarray(tr['traj']); modes=np.linalg.svd(a,full_matrices=False)[2][:2]; y=np.c_[tr['y'],a@modes.T/np.sqrt(a.shape[1])]; y=(y-y.mean(0))/max(1e-8,np.std(y[:,:2])); y[:,2]*=2
    blocks=_bank(x,study)
    if mode==0: ids=[0]
    elif mode==1: ids=[0,1,2,3]
    else: ids=[0,4,5,6]
    z=np.c_[np.ones(len(x)),*[blocks[i] for i in ids]]; z=(z-np.r_[0,z[:n,1:].mean(0)])/np.r_[1,np.maximum(.03,z[:n,1:].std(0))];
    rng=np.random.default_rng(123); found=[]
    for i in range(12):
        d=4 if i<9 else 6
        c=np.zeros((z.shape[1],d)); c[1:5,:4]=(np.eye(4) if i in (0,3,6,9) else np.linalg.qr(rng.normal(size=(4,4)))[0])*[.2,.35,.55][i%3]; c+=rng.normal(0,.02,c.shape)
        l,co=_fit(z[:n],y,c,0.0015 if mode==0 else 0.003,180)
        found.append((l,z@co))
    return sorted(found,key=lambda q:q[0])
