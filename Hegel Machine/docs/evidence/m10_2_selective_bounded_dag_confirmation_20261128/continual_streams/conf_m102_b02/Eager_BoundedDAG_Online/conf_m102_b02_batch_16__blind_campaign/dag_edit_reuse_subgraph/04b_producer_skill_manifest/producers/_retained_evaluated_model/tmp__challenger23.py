"""Fit reference-panel coordinates and learn a compact spatial representation."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.linalg import solve

def D(x): return np.c_[np.ones(len(x)),x]
def fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def cyc(z): return z*np.roll(z,-1,axis=1)

def coordinates(s):
    p=s['panel_a']; u,v=p['u'],p['v']; x=np.r_[s['train']['x'],s['test']['x']]
    A=D(np.c_[u,cyc(u),u**3]); keep=np.ones(len(u),bool)
    for _ in range(12):
        w=fit(A[keep],v[keep]); e=np.linalg.norm(A@w-v,axis=1); new=e<max(.001,3*np.median(e))
        if new.sum()<18 or np.array_equal(new,keep): break
        keep=new
    u,v=u[keep],v[keep]
    def form(z,q): return D(np.c_[z,np.sin(q[0]*z),z*z])
    def err(q):
        a=form(u,q); return (a@fit(a,v)-v).ravel()
    q=least_squares(err,[2.15],bounds=(1,4)).x; w=fit(form(u,q),v)
    z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
    for _ in range(14):
        j=w[1:5]+(q[0]*np.cos(q[0]*z))[:,:,None]*w[5:9]+2*z[:,:,None]*w[9:13]
        z-=np.clip(np.linalg.solve(j.transpose(0,2,1),(form(z,q)@w-x)[...,None])[...,0],-.3,.3)
    p=s['panel_b']; H=lambda a:D(np.sign(a)*np.maximum(abs(a)-.35,0)**1.5); wb=fit(H(p['u']),p['v']); B=lambda a:H(a)@wb
    p=s['panel_c']; u=p['u']; wc=fit(D(cyc(u)),p['v1']); C=lambda a:D(cyc(a))@wc
    def g(a,q): return D(a+q[1]*a*a/(1+q[0]*a*a))
    def e(q):
        a=g(u,q); return (a@fit(a,p['v2'])-p['v2']).ravel()
    q=least_squares(e,[.25,.45],bounds=([0,-3],[8,3])).x; we=fit(g(u,q),p['v2']); E=lambda a:g(a,q)@we
    return z,B,C,E

def features(z,B,C,E):
    b=B(z); c=C(z); e=E(z)
    basic=[z,b,c,e,B(c),B(e),C(b),E(b),C(e),E(c)]
    f=basic.copy()
    for x in [z,b,c,e,z+b,z-b,b-.5*z,c+e,c-e,b+c,b-c,b+e,b-e,z+c,z-c,z+e,z-e]:
        f.extend([x*x,cyc(x),x*np.roll(x,2,axis=1)])
        for a in [.65,1.25,2.15]:
            f.extend([np.sin(a*x)-a*x,np.tanh(a*x)-a*x,x/(1+(a*x)**2)-x,np.arctan(a*x)**2])
    for i,a in enumerate(basic[:4]):
        for b1 in basic[i:4]:
            f.append((a[:,:,None]*b1[:,None,:]).reshape(len(z),-1))
    return np.column_stack(f)

def head(F,Y):
    n=len(Y); A=F[:n]; mean=A.mean(0); scale=np.maximum(A.std(0),.015); X=(F-mean)/scale; A=X[:n]; m=Y.mean(0); y=Y-m
    # Sparse Bayesian regression, with shared scale evidence within each response.
    result=[]
    for k in range(y.shape[1]):
        t=y[:,k]; means=[]
        for power in [0.,.5,1.]:
            a=scale**power; a/=np.median(a); Q=A*a; beta=np.ones(Q.shape[1])/Q.shape[1]; noise=max(.0001*np.var(t),1e-9)
            for _ in range(60):
                W=Q*beta; inv=solve(W@Q.T+noise*np.eye(n),np.c_[t,Q],assume_a='pos'); w=beta*(Q.T@inv[:,0]); lev=np.sum(Q*inv[:,1:],axis=0); proposed=abs(w)/np.sqrt(np.maximum(lev,1e-10)); beta=np.maximum(1e-11,.35*beta+.65*proposed)
            means.append(X@(a*w)+m[k])
        result.append(np.mean(means,axis=0))
    return np.column_stack(result)

from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve

def banks(z,B,C,E):
    b,c,e=B(z),C(z),E(z)
    q={'z':z,'b':b,'c':c,'e':e}
    for nm,x in list(q.items()):
        if nm!='z':
            for name,func in [('b',B),('c',C),('e',E)]:q[name+nm]=func(x)
    for nm,x in [('z',z),('b',b),('c',c),('e',e)]:
        for nn,y in [('z',z),('b',b),('c',c),('e',e)]:
            if nm<nn:
                for a in [-1.,1.]:
                    x1=x+a*y; k=f'{nm}{a:+}{nn}';q[k]=x1
                    for name,func in [('b',B),('c',C),('e',E)]:q[name+k]=func(x1)
    g={}
    for nm,x in q.items():
        i,j=np.triu_indices(x.shape[1],1)
        g[nm+'lin']=x
        g[nm+'quad']=np.c_[x*x,x[:,i]*x[:,j]]
        g[nm+'cyc']=cyc(x)
        for a in [.65,1.25,2.15]:
            for k,h in [('tanh',np.tanh(a*x)),('erf',erf(a*x)),('sin',np.sin(a*x)),('rat',x/(1+(a*x)**2)),('th',np.sign(x)*np.maximum(abs(x)-.22,0)**1.25)]:
                g[nm+k+str(a)]=h
                if a==1.25:g[nm+k+'2']=h*h
    for i,(nm,x) in enumerate(list(q.items())[:13]):
        for nn,y in list(q.items())[i+1:13]:
            g[nm+'*'+nn]=(x[:,:,None]*y[:,None,:]).reshape(len(z),-1)
    return g

def covs(g,n):
    groups=[];names=[]
    for name,F in g.items():
        F=F-F[:n].mean(0); scale=np.maximum(F[:n].std(0),.008); F=F/np.sqrt(scale); K=F@F.T
        K/=max(np.mean(K.diagonal()[:n]),1e-10);groups.append(K);names.append(name)
    # Broad smooth components as a fallback.
    for name in ['z','b','c','e','bz','cb','eb','ez','cz','bz-1.0e']:
        if name+'lin' not in g:continue
        x=g[name+'lin']; x=(x-x[:n].mean(0))/np.maximum(x[:n].std(0),.01)
        dot=x@x.T/x.shape[1]; dist=np.maximum(dot.diagonal()[:,None]+dot.diagonal()-2*dot,0)
        for a in [.08,.3,1.2]:groups.append(np.exp(-a*dist));names.append(name+'rbf'+str(a))
    return names,np.array(groups)

def regress(K,Y,verbose=False,names=None):
    n=len(Y); avg=Y.mean(0); sd=np.maximum(Y.std(0),1e-9); Y=(Y-avg)/sd
    flat=np.ascontiguousarray(K[:,:n,:n].reshape(len(K),-1)); cross=np.ascontiguousarray(K[:,n:,:n].reshape(len(K),-1)); eye=np.eye(n); ps=[]
    for t in Y.T:
        def obj(w):
            V=(w[:-1]@flat).reshape(n,n)+(w[-1]+1e-7)*eye
            c=cho_factor(V,check_finite=False); a=cho_solve(c,t,check_finite=False); W=cho_solve(c,eye,check_finite=False)-a[:,None]*a[None,:]
            return np.log(c[0].diagonal()).sum()+.5*(t@a), np.r_[.5*(flat@W.ravel()),.5*W.trace()]
        pp=[]
        for penalty in [0.,4.]:
            def fn(w):
                l,d=obj(w);return l+penalty*np.sum(w[:-1]),d+np.r_[np.full(len(K),penalty),0.]
            init=np.r_[np.full(len(K),1./len(K)),.015]; w=minimize(fn,init,jac=True,method='L-BFGS-B',bounds=[(0,15.)]*len(K)+[(.000025,1.)],options={'maxiter':145,'ftol':4e-8}).x
            if verbose and penalty==2.:print('groups',[(names[i],round(w[i],3)) for i in np.argsort(w[:-1])[-12:][::-1] if w[i]>.0001], 'noise',w[-1],flush=True)
            V=(w[:-1]@flat).reshape(n,n)+(w[-1]+1e-7)*eye; cross1=(w[:-1]@cross).reshape(-1,n); a=solve(V,t,assume_a='pos'); pp.append(np.r_[t-w[-1]*a,cross1@a]);
        ps.append(np.mean(pp,axis=0))
    return np.column_stack(ps)*sd+avg

def extra_groups(g,z,B,C,E):
    b,c,e=B(z),C(z),E(z)
    vals=[z,b,c,e,b-.5*z,b-z,b+z,b+c,b-c,b+e,b-e,e+c,e-c]
    for i,x in enumerate(vals):
        opp=np.roll(x,2,axis=1); nxt=np.roll(x,-1,axis=1)
        for j,v in enumerate([opp,nxt]):
            for s in [-1,1]:
                u=x+s*v
                for k,a in enumerate([.65,1.25,2.15]):
                    h=np.tanh(a*u); r=u/(1+(a*u)**2)
                    g[f'p{i},{j},{s},{k}']=np.c_[h,r,h*h,r*r]
            g[f'triple{i},{j}']=x*v*np.roll(x,j+1,axis=1)
        for k,a in enumerate([.65,1.25,2.15]):
            h=np.arcsinh(a*x); r=x/(1+(a*x)**2)
            g[f'quot{i},{k}']=np.c_[h,h*h,x*h,r,r*r]
    for i,x in enumerate(vals[:4]):
        for j,v in enumerate(vals[:4]):
            if i!=j:
                g[f'mix{i},{j}']=np.c_[x/(1+1.25*v*v),x/(1+2.15*v*v),x*np.tanh(1.25*v),x*x*v]
    return g

def predict_local(s,variant):
    z,B,C,E=coordinates(s); tr=s['train']; t=s['eval_times']; T=tr['traj'].T
    def basis(q): return -np.expm1(-t[:,None]*q)
    def err(q):
        a=basis(q); return (a@fit(a,T)-T).ravel()
    a=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)
    modes=fit(a,T).T; n=len(modes); rotation=np.linalg.svd(a,full_matrices=False)[2].T
    target=np.c_[tr['y'],modes@rotation]
    g=banks(z,B,C,E)
    if variant==2:g=extra_groups(g,z,B,C,E)
    names,K=covs(g,n)
    p=regress(K,target)
    p[:,1:]=p[:,1:]@rotation.T
    return p,modes,z,B(z)

def run_node(inputs: dict) -> dict:
    p,modes,z,b=predict_local(inputs['__input__'],VARIANT)
    n=len(modes)
    scale=np.maximum(modes.std(0),1e-5)
    return {'y_hat':p[n:,0].copy(),'trajectory_design':np.c_[p[:,1:]/scale,.035*z,.035*b]}

VARIANT=1
