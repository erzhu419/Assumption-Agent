"""Predict scalar and trajectory outputs from calibrated measurements."""
import numpy as np
from scipy.optimize import least_squares
def D(x):return np.c_[np.ones(len(x)),x]
def fit(F,y):return np.linalg.lstsq(F,y,rcond=1e-10)[0]
def cal(s,x):
 p=s['panel_a']; F=D(p['u']); y=p['v']; keep=np.ones(len(F),bool)
 for _ in range(8):
  c=fit(F[keep],y[keep]); e=np.linalg.norm(F@c-y,axis=1);keep=e<max(.001,3*np.median(e))
 return np.linalg.solve(c[1:].T,(x-c[0]).T).T,c,e
def AF(u): return D(np.c_[u,u*np.roll(u,-1,axis=1),u**3])
def coords(s,x):
 p=s['panel_a']; u,v=p['u'],p['v']; F=AF(u); _,_,e=cal(s,x); w=fit(F[e<2],v[e<2]); z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
 for j in range(12):
  J=np.broadcast_to(w[1:5],(len(z),4,4)).copy(); K=w[5:9]; rr=np.roll(z,-1,axis=1)
  J+=rr[:,:,None]*K[None,:,:]+np.roll(z[:,:,None]*K[None,:,:],1,axis=1)+3*z[:,:,None]**2*w[None,9:13,:]
  dz=np.linalg.solve(J.transpose(0,2,1),(AF(z)@w-x)[...,None])[...,0];z-=np.clip(dz,-.4,.4)
 return z

def parts(study):
    tr=study['train']; z=coords(study,np.r_[tr['x'],study['test']['x']]); p=study['panel_b']; u=p['u'];v=p['v']
    def H(x,q):return D(np.sign(x)*np.maximum(abs(x)-q[0],0)**q[1])
    def e(q):
        f=H(u,q);return (f@fit(f,v)-v).ravel()
    q=least_squares(e,[.35,1.5],bounds=([.15,.6],[.6,2.5])).x
    b=H(z,q)@fit(H(u,q),v)
    return z,b

def times(study):
    t=study['eval_times']; T=study['train']['traj']
    def F(p):return -np.expm1(-t[:,None]*p)
    def err(p):
        f=F(p);return (f@fit(f,T.T)-T.T).ravel()
    return F(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)

def kernel(x, degree, decay):
    dot=x@x.T/x.shape[1]
    dist=np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0)
    return (1+dot)**degree+np.exp(-decay*dist)

def predict(study):
    z,b=parts(study)
    f=times(study)
    tr=study['train']; n=len(tr['x'])
    modes=fit(f,tr['traj'].T).T
    sym=np.c_[.3*z+b,.3*z-b]
    sep=np.c_[.3*z,b]
    designs=(np.arcsinh(sym),np.arcsinh(sep),sym,sep)
    def head(spec,target):
        form,degree,decay,ridge=spec
        k=kernel(designs[form],degree,decay)
        return k[n:,:n]@np.linalg.solve(k[:n,:n]+ridge*np.eye(n),target)
    def blend(a,b,w,target):
        pred=head(a,target)
        return pred if w==1 else w*pred+(1-w)*head(b,target)
    y=blend(*((0, 2, 1.4, 1e-06), (2, 3, 0.15, 1e-06), 0.5),tr['y'])
    state=blend(*((1, 3, 0.15, 1e-06), (1, 3, 1.4, 1e-06), 0.25),modes)@f.T
    return y,state


def _models(s, z):
    p=s['panel_b']; h=lambda a: np.sign(a)*np.maximum(abs(a)-.35,0)**1.5
    wb=fit(D(h(p['u'])),p['v']); b=D(h(z))@wb
    p=s['panel_c']; u=p['u']; q=lambda a:a*np.roll(a,-1,axis=1)
    c=fit(D(q(u)),p['v1']); e=fit(D(np.c_[u,np.cos(1.2726*u)]),p['v2']);
    # Include both measured operator order and alternate orderings of mixing and activation.
    az=z@wb[1:]; cz=D(q(z))@c; ez=D(np.c_[z,np.cos(1.2726*z)])@e
    g=[z,b,h(z),az,h(az),q(z),q(b),ez,cz,D(q(b))@c,D(np.c_[b,np.cos(1.2726*b)])@e]
    for k in [-1.25,-.7,-.22,.22,.7,1.25]:
        m=z+k*b
        g.append(np.column_stack([m,h(m),q(m),D(q(m))@c]))
    for k in [-1.,1.]:
        m=np.column_stack([z[:,j]+k*z[:,i] for j in range(4) for i in range(j+1,4)])
        g.append(m)
    return g


def _kernels(g,n):
    out=[]; costs=[]
    def add(k,c):
        scale=np.mean(k.diagonal()[:n]);
        if np.isfinite(scale) and scale>1e-10: out.append(k/scale); costs.append(c)
    def gram(a): return (a@a.T)/a.shape[1]
    def dist(a):
        r=(a*a).sum(1); return np.maximum(r[:,None]+r[None,:]-2*(a@a.T),0)/a.shape[1]
    zz=[]
    for j,a in enumerate(g):
        a=(a-a[:n].mean(0))/np.maximum(a[:n].std(0),.04)
        zz.append(a); cost=1 if j<5 else 1.6
        add(gram(a),cost)
        add(gram(a)**2,cost*1.4)
        r=np.abs(a[:,None]-a[None,:]);
        for sc in [.48,1.7]:
            u=sc*r; Ka=(1+u+u*u/3)*np.exp(-u)
            add(Ka.mean(2),cost)
            if j<5:
                for k in range(a.shape[1]):add(Ka[:,:,k],cost*1.4)
        if j<11:
            for sc in [.12,.7]:add(np.exp(-sc*dist(a)),cost)
    for a in [zz[0],np.c_[zz[0],zz[1]],np.c_[zz[0],zz[2]],np.c_[zz[0],zz[1],zz[5]],np.c_[zz[0],zz[6],zz[7]]]:
        for aa in [a,np.arcsinh(a)]:
            k=gram(aa)
            for deg in [2,3,4]: add((1+k)**deg,1.)
            r=dist(aa)
            for sc in [.08,.3,1.]:
                add(np.exp(-sc*r),1.); add((1+sc*r)**-2,1.)
    add(np.ones_like(out[0]),1.)
    return np.asarray(out),np.asarray(costs)


def _gp_models(K,cost,y):
    from scipy.linalg import cho_factor,cho_solve
    from scipy.optimize import minimize
    n=len(y); s=max(y.std(),.001); y=y/s; mean=y.mean(); y=y-mean
    A=K[:,:n,:n].copy().reshape(len(K),-1); B=K[:,n:,:n].copy().reshape(len(K),-1)
    eye=np.eye(n); out=[]; aa=[]
    def infer(w,noise,loo=False):
        M=(w@A).reshape(n,n)+noise*eye
        L=cho_factor(M,lower=True,check_finite=False); a=cho_solve(L,y,check_finite=False)
        inv=cho_solve(L,eye,check_finite=False)
        if loo: return np.mean(abs(a/inv.diagonal())), ((w@B).reshape(-1,n)@a+mean)*s
        return L,a,inv
    from scipy.optimize import minimize_scalar
    # Marginal likelihood regularization keeps selection inside the training slice.
    for reg in [.15,1.,4.]:
        def obj(theta):
            w=theta[:-1]; v=np.exp(theta[-1]); L,a,inv=infer(w,v)
            G=.5*(inv-a[:,None]*a[None,:]); penalty=reg*cost
            return .5*y@a+np.log(L[0].diagonal()).sum()+w@penalty, np.r_[A@G.ravel()+penalty,np.trace(G)*v]
        initial=np.r_[np.full(len(K),.8/len(K)),np.log(.004)]
        res=minimize(obj,initial,method='L-BFGS-B',jac=True,bounds=[(0,20)]*len(K)+[(-12,-1)],options={'maxiter':180,'ftol':2e-8,'maxcor':12})
        err,pred=infer(res.x[:-1],np.exp(res.x[-1]),True); out.append(pred);aa.append(err)
    # Independent kernel fits provide a stable alternative to optimizing many variance components.
    inds=list(range(max(len(K)-61,0),len(K)-1)); vals=[]
    linear=np.zeros(len(K)); linear[0]=.25;linear[4]=.07;linear[8]=.07;linear[-1]=.2
    for j in inds:
        for mode in [0,1]:
            w=linear.copy(); w[j]+=1.; w[:8]*=1 if mode else .15
            for noise in [3e-5,.0005,.005,.025]:
                err,pred=infer(w,noise,True); vals.append((err,pred))
    vals.sort(key=lambda t:t[0]); err=np.array([x[0] for x in vals[:18]]); weights=np.exp(-30*(err/err[0]-1));
    out.append(weights@np.array([x[1] for x in vals[:18]])/weights.sum());aa.append(err[0])
    return np.array(out),np.array(aa)


def _candidate_predictions(s):
    tr=s['train']; n=len(tr['x']); x=np.r_[tr['x'],s['test']['x']]; z=coords(s,x); g=_models(s,z); K,cost=_kernels(g,n)
    T=np.asarray(tr['traj']); v=np.linalg.svd(T,full_matrices=False)[2][:3];
    y,ey=_gp_models(K,cost,tr['y']); t,et=_gp_models(K,cost,T@v[0]);
    # The two smaller time modes are fitted with a strongly regularized smooth model.
    a=g[0]; b=g[1]; F=np.c_[a,b]; F=F/np.maximum(F[:n].std(0),.05); G=1+F@F.T/F.shape[1]; G=G**2; U=G[n:,:n]@np.linalg.solve(G[:n,:n]+.12*np.eye(n),T@v[1:].T); correction=U@v[1:];
    return y,t[:,:,None]*v[0]+correction,ey,et


def solve_batch(study:dict)->dict:
    y,T,ey,et=_candidate_predictions(study)
    def blend(p,e):
        w=np.exp(-8*(e/e.min()-1)); return np.tensordot(w/w.sum(),p,axes=1)
    y=blend(y,ey); T=blend(T,et); end=T[:,-1]
    return {'y_hat':y,'state_pred':T,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
