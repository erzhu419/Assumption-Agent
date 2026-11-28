"""Calibrated composed operators with separate scalar and trajectory readouts."""
import numpy as np

def D(x): return np.c_[np.ones(len(x)), x]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def P(x): return x*np.roll(x,-1,axis=1)
def AF(x): return D(np.c_[x,P(x),x**3])
def B(x): return D(np.sign(x)*np.maximum(abs(x)-.35,0)**1.5)
def E(x): return D(np.sqrt(1+.924227*x*x)+.831839*x)

def features(s,x):
    a=s['panel_a']; u,v=a['u'],a['v']; F=AF(u); k=np.ones(len(u),bool)
    for _ in range(8):
        w=fit(F[k],v[k]); r=np.linalg.norm(F@w-v,axis=1); h=r<max(.002,3*np.median(r))
        if h.sum()<F.shape[1] or np.array_equal(h,k): break
        k=h
    z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
    for _ in range(14):
        J=w[1:5]+np.roll(z,-1,axis=1)[:,:,None]*w[5:9]+np.roll(z[:,:,None]*w[5:9],1,axis=1)+3*z[:,:,None]**2*w[9:]
        dz=np.linalg.solve(J.transpose(0,2,1),(AF(z)@w-x)[...,None])[...,0]
        z-=np.clip(dz,-.4,.4)
    a=s['panel_b']; b=B(z)@fit(B(a['u']),a['v']); a=s['panel_c']; C=fit(D(P(a['u'])),a['v1']); W=fit(E(a['u']),a['v2']); e=E(z)@W
    return z,b,e,C,W

def kernel(x,n,degree,rbf,reg,y):
    x=x-x[:n].mean(0); x=x/max(.025,np.sqrt(np.mean(x[:n]**2))); L=x@x.T/x.shape[1]
    d=np.maximum(L.diagonal()[:,None]+L.diagonal()-2*L,0); K=(1+L)**degree
    if rbf: K+=np.exp(-rbf*d)
    return K[n:,:n]@np.linalg.solve(K[:n,:n]+reg*np.eye(n),y)

def solve_batch(study:dict)->dict:
    tr=study['train']; n=len(tr['x']); z,b,e,C,W=features(study,np.r_[tr['x'],study['test']['x']]);
    options=_scalar_candidates(z,b,e,C,W,tr['y'])
    y=0.5*(0.5*options['original']+0.5*options['blend30'])+0.5*_structured_y(z,b,C,tr['y'],'projsqp',0.1)
    state=kernel(np.c_[D(P(.5*b))@C,.22*z],n,2,0,.0005,tr['traj'])
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}

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

from scipy.optimize import least_squares, minimize_scalar

def _structured_y(z,b,C,y,mode,pen):
    n=len(y); p=P(b); c=D(p)@C
    def scale(v):
        v=v-v[:n].mean(0)
        return v/np.maximum(.035,v[:n].std(0))
    z,p,c,b=map(scale,(z,p,c,b)); X=np.c_[z,p]; A=D(X); L=A[:n]; rng=np.random.default_rng(107)
    if mode in ('rank1','rank2','fullrank1','fullrank2'):
        full=mode.startswith('full'); d=8 if full else 4; rank=int(mode[-1]); U=X if full else z; V=X if full else p
        def prediction(q,A,U,V):
            a=q[9:9+d*rank].reshape(d,rank); h=q[9+d*rank:].reshape(d,rank)
            return A@q[:9]+np.sum((U@a)*(V@h),axis=1)
        def loss(q):return np.r_[prediction(q,L,U[:n],V[:n])-y,pen*q[9:]]
        def jac(q):
            a=q[9:9+d*rank].reshape(d,rank); h=q[9+d*rank:].reshape(d,rank)
            return np.r_[np.c_[L,(U[:n,:,None]*(V[:n]@h)[:,None,:]).reshape(n,-1),(V[:n,:,None]*(U[:n]@a)[:,None,:]).reshape(n,-1)],np.c_[np.zeros((2*d*rank,9)),pen*np.eye(2*d*rank)]]
        q0=fit(L,y); sol=min((least_squares(loss,np.r_[q0,rng.normal(0,.3,2*d*rank)],jac=jac,max_nfev=160) for _ in range(6)),key=lambda r:r.cost)
        return prediction(sol.x,A[n:],U[n:],V[n:])
    if mode.startswith('proj'):
        base=D(np.c_[z,p]); kind=mode[4:]; V=p if kind[-1]=='p' else c; kind=kind[:-1]
        fs={'sin':np.sin,'asinh':np.arcsinh,'tanh':np.tanh,'rat':lambda x:x/(1+x*x),'sq':np.square}
        f=fs[kind]
        def build(q):
            h=f(q[0]*V+q[1]); h=scale(h); return np.c_[base,h,(z[:,:,None]*h[:,None,:]).reshape(len(z),-1)]
        def fitq(q,ret=False):
            H=build(q); F=H[:n]; ridge=np.r_[np.full(9,.004),np.full(4,pen*.5),np.full(16,pen)]; coef=np.linalg.solve(F.T@F+np.diag(ridge),F.T@y)
            return H[n:]@coef if ret else np.r_[F@coef-y,np.sqrt(ridge)*coef,.01*q]
        res=min((least_squares(fitq,[k,s],bounds=([.03,-3],[6,3]),max_nfev=65) for k in (.35,1.5,3.5) for s in (0.,.8)),key=lambda r:r.cost)
        return fitq(res.x,True)
    raise ValueError(mode)
