import numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize, nnls

def _gram(a):
    return a@a.T/a.shape[1]

def _poly_parts(z,b,p):
    return [z,b,p]+[np.column_stack([v[:,i]*v[:,j] for i,j in combinations_with_replacement(range(v.shape[1]),2)]) for v in (z,b,p)]+[(u[:,:,None]*v[:,None,:]).reshape(len(z),-1) for u,v in ((z,b),(z,p),(b,p))]

def _scalar_candidates(z,b,e,C,W,y):
    n=len(y); p=P(b); c=D(p)@C; h=D(P(b+.5*e))@C; ep=E(b)@W
    candidates=[]
    cv=[]
    names=[]
    def add(K,key,reg,transform=0):
        K0=K[:n,:n].copy(); K0.flat[::n+1]+=reg
        ch=cho_factor(K0,check_finite=False)
        target=y if transform==0 else np.arcsinh(transform*y)/transform
        w=cho_solve(ch,target,check_finite=False); inv=np.diag(cho_solve(ch,np.eye(n),check_finite=False)); out=K[n:,:n]@w; lo=target-w/inv
        if transform: out=np.sinh(np.clip(out*transform,-4,4))/transform; lo=np.sinh(np.clip(lo*transform,-4,4))/transform
        candidates.append(out);cv.append(lo);names.append(key)
    def centered(x):return x-x[:n].mean(0)
    def gd(x):
        a=centered(x); a=a/max(.02,np.sqrt(np.mean(a[:n]**2))); L=_gram(a); d=np.maximum(L.diagonal()[:,None]+L.diagonal()-2*L,0); return L,d
    for s in (0,.2,.4,.6,.9):
      for v in (e,ep,z):
        q=D(P(b+s*v))@C
        for az in (.12,.22,.4,.65):
          for ab in (0,.35,.8):
            f=np.c_[az*z,ab*b,q]; L,d=gd(f)
            for k in (.12,.5,1.3):
              add((1+L)**2+np.exp(-k*d),('cascade',s,az,ab,k),.00002)
    # Explicitly separate the linear, bilinear and quartic blocks rather than
    # shrinking each coefficient at the same rate.
    Q=_poly_parts(z,b,p); G=[]
    for q in Q:
        q=centered(q); G.append(_gram(q)/max(.0001,np.mean(q[:n]**2)))
    z0,b0,p0,zz,bb,pp,zb,zp,bp=G
    for w0 in (.08,.3,1.):
      for w1 in (.015,.1,.45,1.5):
       for w2 in (.01,.1,.5):
        for w3 in (0.,.025,.15):
         K=1+z0+w0*p0+w1*zp+w2*(zz+pp)+w3*(b0+bb+zb+bp)
         for rg in (.00015,.002,.014): add(K,('block',w0,w1,w2,w3,rg),rg)
    for f in (np.c_[z,p],np.c_[z,p,b],np.c_[z,p,h],np.c_[z,p,ep]):
      for size in (2,3):
        X=np.column_stack([f]+[np.column_stack([np.prod(f[:,ids],axis=1) for ids in combinations_with_replacement(range(f.shape[1]),d)]) for d in range(2,size+1)])
        X=centered(X); X/=np.maximum(.02,np.sqrt(np.mean(X[:n]**2,0)))
        # Sparse empirical Bayes: diagonal coefficient variances with a stable noise floor.
        for floor in (.00008,.001,.004):
            V=X[:n]; w=np.full(X.shape[1],.025); residual=y-y.mean();
            for _ in range(240):
                K=(V*w)@V.T; K.flat[::n+1]+=floor
                ch=cho_factor(K,check_finite=False); A=cho_solve(ch,V,check_finite=False); mu=w*(A.T@residual); leverage=w*np.sum(V*A,0); nw=np.maximum(1e-8,mu*mu/np.maximum(.04,leverage)); nw=np.minimum(nw,4.); w=.64*w+.36*nw
            add(1+(X*w)@X.T,('sparse',f.shape[1],size,floor),floor)
    L,d=gd(np.c_[h,.22*z,b]); add((1+L)**2+np.exp(-.7*d),('original',),1e-6)
    pred=np.array(candidates); cv=np.array(cv); score=np.mean(abs(cv-y),axis=1); idx=np.argsort(score); out={}
    for family in ('cascade','block','sparse','original'):
       ii=np.array([i for i in idx if names[i][0]==family]);
       out[family]=pred[ii[:min(12,len(ii))]].mean(0)
    for k in (8,30,100):out['blend'+str(k)]=pred[idx[:k]].mean(0)
    # A convex, regularized ensemble of the best distinct leave-one-out predictors.
    ix=np.array([i for fam in ('cascade','block','sparse') for i in [j for j in idx if names[j][0]==fam][:15]]); V=cv[ix].T
    w=nnls(np.r_[V,np.full((1,len(ix)),30.),.035*np.eye(len(ix))],np.r_[y,30.,np.zeros(len(ix))])[0]; out['stack']=w@pred[ix]/max(w.sum(),1e-8)
    return out
