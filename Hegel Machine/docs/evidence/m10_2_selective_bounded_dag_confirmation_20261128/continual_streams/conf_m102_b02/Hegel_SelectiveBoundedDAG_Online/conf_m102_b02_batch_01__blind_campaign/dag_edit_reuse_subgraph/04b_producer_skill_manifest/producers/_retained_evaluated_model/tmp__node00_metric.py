"""Calibrate latent inputs and fit a smooth metric by training-only leave-one-out prediction."""
import numpy as np
from scipy.optimize import minimize


def _latent(study):
    p = study['panel_a']
    u, v = np.asarray(p['u'],float), np.asarray(p['v'],float)
    d = np.c_[np.ones(len(u)),u]
    keep = np.ones(len(u),bool)
    for _ in range(8):
        c = np.linalg.lstsq(d[keep],v[keep],rcond=None)[0]
        e = np.linalg.norm(d@c-v,axis=1)
        k = e <= max(.001,3*np.median(e))
        if np.array_equal(k,keep) or k.sum()<d.shape[1]: break
        keep=k
    c=np.linalg.lstsq(d[keep],v[keep],rcond=None)[0]
    x,xt=(np.asarray(study[k]['x'],float) for k in ('train','test'))
    return np.linalg.solve(c[1:].T,(np.vstack((x,xt))-c[0]).T).T,len(x)


def _features(u,kind):
    out=[np.ones((len(u),1)),u]
    if kind>=1: out.extend([u*u,np.tanh(2*u)-u])
    if kind>=2: out.append(np.column_stack([u[:,i]*u[:,j] for i in range(u.shape[1]) for j in range(i+1,u.shape[1])]))
    return np.column_stack(out)


def _coordinate(F,p,shape):
    W=p[:-2].reshape(shape); t=F@W
    f=np.tanh(2*t); g=np.cos(2*t)-1
    z=t+p[-2]*(f-t)+p[-1]*g
    return z,f,g,t


def _metric(F,Y,initial,kind):
    n,d=initial.shape; p0=np.r_[initial.ravel(),0.,0.]; ident=np.eye(len(F)); lam=2e-4 if kind!=2 else 1e-3
    def loss(p):
        z,f,g,t=_coordinate(F,p,(n,d)); h=np.arcsinh(3.5*z); hh=h*h; zh=z*h
        l=z@z.T; c=h@h.T; q=c+.25*l
        K=1+l+c+hh@hh.T+zh@zh.T+.2*q*q
        H=np.linalg.solve(K+(1.e-6*np.trace(K)/len(K))*ident,ident)
        a=H@Y; diag=np.diag(H); e=a/diag[:,None]
        # Smooth absolute loss, balanced across all response modes.
        r=np.sqrt(e*e+.012**2); b=e/r/diag[:,None]/e.size
        v=b@Y.T; v[np.diag_indices_from(v)]-=np.sum(b*a,axis=1)
        v=-H@((v+v.T)*.5)@H; v[np.diag_indices_from(v)]+=1.e-6*np.trace(v)/len(v)
        a1=(2*v)@zh
        dz=(2*v*(1+.1*q))@z+a1*h
        dh=(2*v*(1+.4*q))@h+((4*v)@hh)*h+a1*z
        dz+=dh*(3.5/np.sqrt(1+(3.5*z)**2))
        dt=dz*(1+p[-2]*(2*(1-f*f)-1)-2*p[-1]*np.sin(2*t))
        delta=p-p0
        gradient=np.r_[(F.T@dt).ravel(),np.sum(dz*(f-t)),np.sum(dz*g)]+2*lam*delta
        return r.mean()+lam*np.dot(delta,delta),gradient
    result=minimize(loss,p0,jac=True,method='L-BFGS-B',bounds=[(-1.5,1.5)]*(len(p0)-2)+[(-1.25,1.25),(-1.,1.)],options={'maxiter':160,'ftol':1e-10})
    return result.fun,result.x


def _fit(study,kind,count):
    u,n=_latent(study); F=_features(u,kind); tr=study['train']; traj=np.asarray(tr['traj'],float)
    T=np.linalg.svd(traj,full_matrices=False)[2][:2]
    Y=np.c_[tr['y'],traj@T.T]; Y=(Y-Y.mean(0))/np.maximum(Y.std(0),.02)
    dim=u.shape[1]; starts=[np.eye(dim)]
    for k,v in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
        p=study[k]; U=np.asarray(p['u']); s=np.c_[np.ones(len(U)),U,np.tanh(2*U)] if v!='v2' else np.c_[np.ones(len(U)),U,np.cos(1.3*U)]
        W=np.linalg.lstsq(s,p[v],rcond=None)[0][1:1+dim]; a,_,b=np.linalg.svd(W); starts.extend([a@b,(a@b).T])
    solutions=[]
    for O in starts[:count]:
        init=np.zeros((F.shape[1],dim)); init[1:1+dim]=.25*O
        risk,p=_metric(F[:n],Y,init,kind)
        z=_coordinate(F,p,init.shape)[0]; solutions.append((risk,z))
    return [z for _,z in sorted(solutions,key=lambda s:s[0])],n


def run_node(inputs: dict) -> dict:
    options,n=_fit(inputs['__input__'],KIND,7)
    z=options[0]
    return {'z':z[:n].copy(),'zt':z[n:].copy()}

KIND=0
