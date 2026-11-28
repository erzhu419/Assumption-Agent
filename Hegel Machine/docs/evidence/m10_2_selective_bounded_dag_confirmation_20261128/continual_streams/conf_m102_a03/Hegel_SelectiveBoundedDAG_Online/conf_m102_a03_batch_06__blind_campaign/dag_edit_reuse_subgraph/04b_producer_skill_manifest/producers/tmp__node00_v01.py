"""Robust reference-panel calibration and positive-definite feature kernels."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _sensor(panel, x):
    u,v=(np.asarray(panel[k],float) for k in ('u','v'))
    d=np.stack((np.ones_like(u),u,np.roll(u,-1,axis=1),u*np.roll(u,-1,axis=1),u**3),axis=2)
    w=np.ones(len(u))
    for _ in range(24):
        c=np.array([np.linalg.lstsq(d[:,j]*w[:,None],v[:,j]*w,rcond=None)[0] for j in range(u.shape[1])]).T
        r=np.linalg.norm(np.einsum('ndp,pd->nd',d,c)-v,axis=1)
        w=np.sqrt(np.minimum(1,1.5*max(np.median(r),1e-12)/np.maximum(r,1e-12)))
    z=(x-c[0])/c[1]
    j=np.arange(u.shape[1]); k=(j+1)%len(j)
    for _ in range(24):
        zn=z[:,k]
        r=c[0]+c[1]*z+c[2]*zn+c[3]*z*zn+c[4]*z**3-x
        J=np.zeros((len(z),len(j),len(j)))
        J[:,j,j]=c[1]+c[3]*zn+3*c[4]*z*z
        J[:,j,k]+=c[2]+c[3]*z
        step=np.linalg.solve(J,r[...,None])[...,0]
        z-=step/np.maximum(1,np.max(abs(step),axis=1,keepdims=True))
    return z


def _basis(z, typ):
    return np.column_stack((np.ones(len(z)),z,{'b':lambda:np.sin(2.8*z),'c':lambda:np.arctan(2.1*z),'q':lambda:np.cos(1.28*z)}[typ]()))


def _groups(study,z):
    B,C,Q=[_basis(z,t)@np.linalg.lstsq(_basis(study[p]['u'],t),study[p][k],rcond=None)[0] for t,p,k in [('b','panel_b','v'),('c','panel_c','v1'),('q','panel_c','v2')]]
    S=np.sin(2.8*z); A=np.arctan(2.1*z); H=np.cos(1.28*z)-1
    fs=[z,S,A,H,B,C,Q]
    G=[np.ones((len(z),len(z)))]; names=['1']
    def add(name,f):
        k=f@f.T; G.append(k/max(np.mean(np.diag(k)),1e-10)); names.append(name)
    for j,f in enumerate(fs): add(str(j),f)
    for i in range(len(fs)):
        for j in range(i,len(fs)):
            f,g=fs[i],fs[j]
            add(f'{i}{j}',(f[:,:,None]*g[:,None,:]).reshape(len(z),-1))
            if j!=i: add(f'{i}d{j}',f*g)
    for j,f in enumerate([z,B,C,Q,np.column_stack((z,S,A,H))]):
        k=f@f.T/f.shape[1]; G.append(k**3/max(np.mean(np.diag(k)**3),1e-10)); names.append(f'cube{j}')
    return np.array(G),names


def _select(G,n,y):
    scale=max(np.sqrt(np.mean(y*y)),1e-10); y=y/scale
    T=G[:,:n,:n].copy().reshape(len(G),-1); eye=np.eye(n)
    def obj(log):
        w=np.exp(log); K=(w[:-1]@T).reshape(n,n)+w[-1]*eye
        c=cho_factor(K,check_finite=False); a=cho_solve(c,y,check_finite=False)
        P=cho_solve(c,eye,check_finite=False)-np.outer(a,a)
        return 0.5*(y@a)+np.log(np.diag(c[0])).sum(), 0.5*w*np.r_[T@P.ravel(),np.trace(P)]
    ini=np.r_[np.full(len(G),-5.),-5.]
    w=np.exp(minimize(obj,ini,jac=True,method='L-BFGS-B',bounds=[(-15,4)]*len(G)+[(-13,0)],options={'maxiter':130}).x)
    K=np.einsum('i,ijk->jk',w[:-1],G[:,:,:n]); return K[:n]+w[-1]*eye,K[n:]


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; train=study['train']; n=len(train['x'])
    z=_sensor(study['panel_a'],np.vstack((train['x'],study['test']['x'])))
    G,names=_groups(study,z)
    Ry,Rt=_select(G,n,np.asarray(train['y'],float))
    Y=np.asarray(train['traj'],float)
    modes=np.linalg.svd(Y,full_matrices=False)[2][:3]
    state=[_select(G,n,Y@mode) for mode in modes]
    return {'readout_train':Ry,'readout_test':Rt,'modes':modes,'state_train':np.stack([k for k,_ in state]),'zt':np.concatenate([k for _,k in state],axis=1)}
