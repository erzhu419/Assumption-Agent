import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize


def _adaptive(D, Y, k, h, debug=False):
    # Empirical-Bayes kernel with separate priors for coordinate blocks,
    # coordinate nonlinearities, pair interactions and full interactions.
    n,m=Y.shape; N=len(D['x']); atoms=[]; names=[]
    def append(name,K):
        atoms.append(K/max(np.mean(np.diag(K)[:n]),1e-10)); names.append(name)
    Z={}
    for key,z in D.items():
        if key not in ('x','b','c','d','bc','bd'): continue
        z=(z-z[:n].mean(0))/np.maximum(z[:n].std(0),.01); Z[key]=z
        dx=(z[:,None,:]-z[None,:,:])**2
        append(key+' lin',z@z.T)
        for power in (2,3):
            p=z**power;p-=p[:n].mean(0);append(key+' uni'+str(power),p@p.T)
        p=np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i+1,4)])
        p-=p[:n].mean(0);append(key+' bilinear',p@p.T)
        for scale in (1.3,3.0,6.0):
            g=np.exp(-dx/(2*scale**2)); append(key+' add'+str(scale),g.mean(2));append(key+' full'+str(scale),g.prod(2))
    for a,b in [('x','b'),('x','c'),('x','d'),('b','c'),('b','d'),('c','d')]:
        p,q=Z[a],Z[b]; f=(p[:,:,None]*q[:,None,:]).reshape(N,-1); f-=f[:n].mean(0);append(a+b+' interactions',f@f.T)
    append('constant', np.ones((N,N))*10);append('noise',np.eye(N)); mats=np.asarray(atoms); tr=mats[:,:n,:n].copy(); cross=mats[:,n:,:n].copy(); tr2=tr.reshape(len(tr),-1); init=np.full(len(mats),np.log(.015)); init[-1]=np.log(.003)
    def gp(Q):
        if Q.ndim==1: Q=Q[:,None]
        mean=Q.mean(0); sd=np.maximum(Q.std(0),1e-6); V=(Q-mean)/sd; p=V.shape[1]; I=np.eye(n)
        def fun(w):
            a=np.exp(w); C=(a@tr2).reshape(n,n); ch=cho_factor(C,lower=True,check_finite=False); alp=cho_solve(ch,V,check_finite=False)
            M=cho_solve(ch,I,check_finite=False)-alp@alp.T/p
            obj=np.log(np.diag(ch[0])).sum()+.5*np.sum(V*alp)/p
            # Weak stabilizing prior, especially important for small panels.
            d=w-init; strength=.023
            return obj+strength*np.dot(d,d),.5*a*(tr2@M.ravel())+2*strength*d
        r=minimize(fun,init,jac=True,method='L-BFGS-B',bounds=[(-13,3)]*(len(init)-1)+[(-13,0)],options={'maxiter':150,'ftol':1e-9}); a=np.exp(r.x)
        if debug: print('kernel',[(names[j],round(a[j],4)) for j in np.argsort(a)[-9:][::-1]],'noise',a[-1])
        C=(a@tr2).reshape(n,n); W=cho_solve(cho_factor(C,lower=True,check_finite=False),V,check_finite=False)
        return ((a@cross.reshape(len(a),-1)).reshape(N-n,n)@W)*sd+mean
    # Learn a joint readout of y and independent temporal modes, plus a
    # separate readout for the dominant temporal direction.
    _,_,basis=np.linalg.svd(Y[:,1:],full_matrices=False); basis=basis[:3]; amps=Y[:,1:]@basis.T
    j=gp(np.c_[Y[:,0],amps]); py=gp(Y[:,0])[:,0]; a=gp(amps); a[:,0]=.55*a[:,0]+.45*gp(amps[:,0])[:,0]
    return np.c_[.65*py+.35*j[:,0],(.65*a+.35*j[:,1:])@basis]
