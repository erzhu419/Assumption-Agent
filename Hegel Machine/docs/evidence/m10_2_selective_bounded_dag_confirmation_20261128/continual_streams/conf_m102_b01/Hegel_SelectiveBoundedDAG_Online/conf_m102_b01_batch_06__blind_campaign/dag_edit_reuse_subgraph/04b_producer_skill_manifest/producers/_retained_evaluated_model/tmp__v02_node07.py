import itertools
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist


def _aff(x):
    return np.column_stack((np.ones(len(x)),x))


def _ring(x):
    return x*np.roll(x,-1,axis=1)


def _cross(x):
    return np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations(range(x.shape[1]),2)])


def _panel(w,p):
    u=np.asarray(p['u']); a=_aff(_ring(u)); v1=_aff(_ring(w))@np.linalg.lstsq(a,p['v1'],rcond=None)[0]
    target=np.asarray(p['v2']); best=[]
    for kind in range(5):
        def design(z,g):
            h={'0':lambda: z**3,'1':lambda: z*z*z/(1+g*z*z),'2':lambda: np.sin(g*z)-g*z,'3':lambda: np.tanh(g*z)-g*z,'4':lambda: z**3/np.sqrt(1+g*z*z)}[str(kind)]()
            return _aff(np.c_[z,z*z,h])
        def objective(g, ret=False):
            a=design(u,g); q=np.linalg.pinv(a,rcond=1.e-10); b=q@target
            r=(a@b-target)/np.maximum(1-np.einsum('ij,ji->i',a,q)[:,None],.04)
            return design(w,g)@b if ret else np.mean(r*r)
        for g in [.2,.7,1.5,3.,6.]:best.append((objective(g),objective(g,True)))
    best.sort(key=lambda v:v[0]); v2=np.mean([q for _,q in best[:4]],axis=0)
    return v1,v2


def _fit(w,y,tr,p, details=False):
    n=len(y); r=_ring(w); v,z=_panel(w,p); d=w-np.roll(w,2,axis=1); e=w+np.roll(w,2,axis=1)
    b=np.c_[y,tr]; cent=b.mean(0); b=b-cent
    def std(a):
        q=a-a[:n].mean(0); return q/np.maximum(q[:n].std(0),1.e-6)
    W,R,V,Z,D,E=map(std,[w,r,v,z,d,e]);
    kw,kr,kv,kz=[q@q.T/q.shape[1] for q in [W,R,V,Z]]
    I=np.eye(n); unit=np.ones((len(w),len(w)))
    base=unit+kw*.4+kr
    families=[]
    def add(label,A):
        A=std(A); k=A@A.T/A.shape[1]; families.append((label,k))
    for g in [.35,1.,2.5,6.]:
        for x,name in [(r,'ring'),(v,'v1'),(z,'v2')]:
            add((name,'nonlin',g), np.c_[np.tanh(g*x),np.sin(g*x),x*x,x**3/(1+g*x*x)])
        for x,name in [(w,'w'),(d,'diff'),(e,'sum'),(r,'ring'),(v,'v1'),(z,'v2')]:
            gate=1/(1+g*x*x)
            for a,an in [(r,'r'),(v,'v'),(np.c_[r,w],'rw')]:
                add((name,an,g), (a[:,:,None]*(gate-1)[:,None,:]).reshape(len(w),-1))
        add(('matched',g),np.c_[v/(1+g*z*z),z/(1+g*v*v),v*z])
    add(('polynomial',),np.c_[w*w,w**3,r*r,_cross(r),(w[:,:,None]*r[:,None,:]).reshape(len(w),-1)])
    add(('panel',),np.c_[z,z*z,v*v,v*z,_cross(v),(v[:,:,None]*z[:,None,:]).reshape(len(w),-1)])
    for X,name in [(np.c_[W,R],'wr'),(np.c_[W,V],'wv'),(np.c_[W,R,Z],'wrz'),(np.c_[V,Z],'vz'),(np.c_[W,R,D,E],'wrd')]:
        dk=cdist(X,X,'sqeuclidean')/X.shape[1]; lk=X@X.T/X.shape[1]
        for g in [.025,.09,.3,1.]:
            addk=[('rbf',np.exp(-g*dk)),('rq',1/(1+g*dk)),('quad',(1+g*lk)**2),('cube',(1+g*lk)**3)]
            for k,f in addk:families.append(((name,g,k),f))
    all_results=[]
    # Each candidate produces exact leave-one-out residuals; no test labels enter selection.
    for label,k in families:
        for ratio in [.13,.7,4.]:
            for ridge in [.001,.01,.065,.25]:
                A=base+ratio*k
                cf=cho_factor(A[:n,:n]+ridge*I,check_finite=False)
                alpha=cho_solve(cf,b,check_finite=False)
                dd=np.diag(cho_solve(cf,I,check_finite=False))
                err=alpha/dd[:,None]; pred=A[n:,:n]@alpha+cent
                all_results.append((label,ratio,ridge,err,pred))
    sv=np.linalg.svd(tr-tr.mean(0),full_matrices=False)[2][0]
    if sv[-1]<0:sv=-sv
    amp=(tr-tr.mean(0))@sv
    beta=(amp@(y-y.mean()))/max(amp@amp,1.e-8)
    ty=np.array([abs(q[3][:,0]).mean() for q in all_results]); tt=np.array([abs(q[3][:,1:]).mean() for q in all_results]); td=np.array([abs(q[3][:,0]-beta*(q[3][:,1:]@sv)).mean() for q in all_results]); ta=np.array([abs(q[3][:,1:]@sv).mean() for q in all_results]);
    def avg(loss,fun):
        ids=np.argsort(loss)[:24]; a=np.exp(-(loss[ids]-loss[ids[0]])/max(.035*loss[ids[0]],1e-6)); return sum(ai*fun(all_results[j]) for ai,j in zip(a,ids))/a.sum()
    yh=avg(ty,lambda q:q[4][:,0]); pred=avg(tt,lambda q:q[4][:,1:]);
    yr=y.mean()+avg(td,lambda q:q[4][:,0]-y.mean()-beta*((q[4][:,1:]-tr.mean(0))@sv))+beta*avg(ta,lambda q:(q[4][:,1:]-tr.mean(0))@sv)
    if details: print('inner minima',ty.min(),tt.min(),td.min(),'beta',beta,'best labels',all_results[ty.argmin()][:3],all_results[tt.argmin()][:3])
    return .3*yh+.7*yr,pred


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; train=study['train']; w=np.asarray(inputs['node_00']['w']);
    y,tr=_fit(w,np.asarray(train['y']),np.asarray(train['traj']),study['panel_c']); tail=tr[:,-1]
    return {'y_hat':y,'state_pred':tr,'diagnostics_vec':np.array([y.mean(),y.std(),tail.mean(),tail.std()])}
