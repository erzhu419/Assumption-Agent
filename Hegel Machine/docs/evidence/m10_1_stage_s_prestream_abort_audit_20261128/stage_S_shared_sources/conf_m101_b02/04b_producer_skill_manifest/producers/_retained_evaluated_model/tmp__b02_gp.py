import importlib.util,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve,lstsq
from scipy.optimize import minimize
sp=importlib.util.spec_from_file_location('base','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def gp(Z,Y,n,kernel,ard):
    Z=(Z-Z[:n].mean(0))/np.maximum(Z[:n].std(0),1e-8)
    Y=np.asarray(Y); mu=Y.mean(0); sd=np.maximum(Y.std(0),1e-8); V=(Y-mu)/sd
    D=(Z[:,None,:]-Z[None,:,:])**2/Z.shape[1]; d=D[:n,:n]; eye=np.eye(n); p=Z.shape[1] if ard else 1
    if not ard: d=d.sum(2,keepdims=True); D=D.sum(2,keepdims=True)
    def kern(d,q):
        w=np.exp(q[:p]); e=d@w
        if kernel=='rbf':
            K=np.exp(-e); G=-K
        else:
            r=np.sqrt(e); K=(1+r+r*r/3)*np.exp(-r); G=-(1+r)*np.exp(-r)/6
        return K,G,w
    def fun(q):
        K,G,w=kern(d,q); v=np.exp(q[p]); C=cho_factor(K+v*eye,check_finite=False); A=cho_solve(C,V,check_finite=False); H=V.shape[1]*cho_solve(C,eye,check_finite=False)-A@A.T
        loss=.5*np.sum(A*V)+V.shape[1]*np.log(np.diag(C[0])).sum()+.35*np.sum(q[:p]**2)
        g=.5*np.r_[np.einsum('ij,ijk->k',H*G,d)*w, v*np.trace(H)]; g[:p]+=.7*q[:p]
        return loss,g
    q=minimize(fun,np.r_[np.zeros(p),-7.],jac=True,method='L-BFGS-B',bounds=[(-6,4)]*p+[(-14,-1.5)],options={'maxiter':45}).x
    K=kern(D,q)[0]; A=cho_solve(cho_factor(K[:n,:n]+np.exp(q[-1])*eye),V)
    return mu+sd*(K[n:,:n]@A)

def trial():
    s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; n=len(x); a=np.arange(n); t=s['eval_times']; Z=m._coordinates(s,x)
    z=(x-x.mean(0))/x.std(0); rad=np.sum(z*z,axis=1)
    splits=[a[56:],*np.array_split(np.random.default_rng(34).permutation(n),4),np.argsort(rad)[-12:]]
    out={}
    for J in splits:
        I=np.setdiff1d(a,J); ii=np.r_[I,J]; f=m._basis(t,T[I]); R=lstsq(f,T[I].T)[0].T
        for path in ('raw','ab','abi','abc','abd','abcd'):
            for k in ('mat','rbf'):
                for ard in (False,True):
                    key=path,k,ard; zz=Z[path][ii]; py=gp(zz,y[I,None],len(I),k,ard)[:,0]; pt=gp(zz,R,len(I),k,ard)@f.T
                    out.setdefault(key,[]).append([abs(py-y[J]).mean(),abs(pt-T[J]).mean(),abs(pt[:,-1]-T[J,-1]).mean()])
    for j in (0,2):
        for key,e in sorted(out.items(),key=lambda kv: np.mean(np.array(kv[1])[:,j]))[:12]: print(j,key,np.round(np.array(e)[:,j],5),flush=True)
if __name__=='__main__':trial()
