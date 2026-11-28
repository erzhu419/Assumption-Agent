import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement

_I = np.arange(4)
_J = (_I + 1) % 4

def _inverse(panel, x):
    u, v = panel['u'], panel['v']
    def basis(a, j):
        b = (j + 1) % 4
        return np.column_stack((np.ones(len(a)), a[:,j], a[:,b], a[:,j]*a[:,b], a[:,j]**3))
    matrices = [basis(u,j) for j in _I]
    weights = np.ones(len(u))
    for _ in range(20):
        coeff = np.array([np.linalg.lstsq(m*weights[:,None], v[:,j]*weights, rcond=None)[0] for j,m in enumerate(matrices)])
        err = np.linalg.norm(v-np.column_stack([m@c for m,c in zip(matrices,coeff)]),axis=1)
        weights = 1/np.hypot(1,err/max(1e-4,1.8*np.median(err)))
    k = np.argsort(err)[:-3]
    coeff = np.array([np.linalg.lstsq(m[k],v[k,j],rcond=None)[0] for j,m in enumerate(matrices)])
    linear = np.diag(coeff[:,1]); linear[_I,_J] = coeff[:,2]
    def residual(a, row):
        return coeff[:,0]+linear@a+coeff[:,3]*a*a[_J]+coeff[:,4]*a**3-row
    return np.array([least_squares(residual,np.linalg.solve(linear,row-coeff[:,0]),args=(row,)).x for row in x])

def _panel_b(panel, x):
    u,v=panel['u'], panel['v']
    def design(u, a):
        return np.c_[np.ones(len(u)),u,np.sign(u)*np.log1p(a[0]*abs(u)**a[1])]
    def residual(a):
        m=design(u,a); p=np.linalg.pinv(m)
        return ((m@(p@v)-v)/(1-(m*p.T).sum(1))[:,None]).ravel()
    a=least_squares(residual,[1.3,3.2],bounds=([.005,1.5],[7,6])).x
    return design(x,a)@np.linalg.lstsq(design(u,a),v,rcond=None)[0]

def _polys(x, degree):
    return np.column_stack([np.ones(len(x))]+[np.prod(x[:,k],axis=1) for d in range(1,degree+1) for k in combinations_with_replacement(range(4),d)])


from scipy.linalg import solve

def _calibrate_b(panel,x,key,extra,theta):
    if key=="baseline": return _panel_b(panel,x)
    u,v=panel["u"],panel["v"]
    i,j=np.triu_indices(4,1)
    def design(x,a):
        fs={
            'csat':lambda x,a:np.tanh(a[0]*x)**a[1] if False else np.sign(x)*np.tanh(a[0]*abs(x))**a[1],
        }
        g=fs[key](x,a)
        extras={"lin":[x],"none":[],"quad":[x,x*x],"cyc":[x,x*np.roll(x,-1,axis=1)],"cross":[x,x[:,i]*x[:,j]],"cube":[x,x**3],"mix":[x,x*x,x*np.roll(x,-1,axis=1)]}
        return np.column_stack([np.ones(len(x)),g]+extras[extra])
    def residual(a):
        F=design(u,a); H=np.linalg.pinv(F,rcond=1e-10)
        return ((v-F@(H@v))/np.maximum(.05,1-(F*H.T).sum(1))[:,None]).ravel()
    lo,hi=([.001,.03],[8,6]) if key not in ("poly","ratmix") else ([-.35,-3],[5,3])
    if key=="ratmix": lo=[.001,-3]
    a=least_squares(residual,theta,bounds=(lo,hi),max_nfev=50).x
    return design(x,a)@np.linalg.lstsq(design(u,a),v,rcond=1e-10)[0]
def pf(b,typ,a,w,c1):
 i,j=np.triu_indices(4,1)
 P=_polys(b,2); b0,b1=b[:,i],b[:,j]; qp=b0*b1; plus=b0+b1; minus=b0-b1
 if typ=='poly':return _polys(b,int(a))
 if typ=='bounded':return np.c_[b,np.tanh(a*b),np.tanh(a*P[:,5:])]
 if typ=='satcubic':return np.c_[P,np.tanh(a*_polys(b,3)[:,15:])/a]
 if typ=='pairshape':return np.c_[P,qp/(1+a*plus**2),qp/(1+a*minus**2),np.sin(a*qp),np.cos(a*qp)]
 if typ=='pairseparable': return np.c_[P,qp/(1+a*(b0*b0+b1*b1)),qp/(1+a*qp**2),qp/(1+a*plus**2),qp/(1+a*minus**2)]
 if typ=='quotient':return np.c_[P,(P[:,:,None]/(1+a*plus[:,None,:]**2)).reshape(len(P),-1)]
 if typ=='cyclequot':return np.c_[P,(P[:,:,None]/(1+a*(b+np.roll(b,-1,axis=1))[:,None,:]**2)).reshape(len(P),-1)]
 if typ=='mixed':return np.c_[P,w,c1,(b[:,:,None]*c1[:,None,:]).reshape(len(P),-1),qp/(1+a*plus**2),qp/(1+a*minus**2)]
 if typ=='axis':return np.c_[P,(b[:,:,None]*np.tanh(a*b)[:,None,:]).reshape(len(P),-1),qp/(1+a*plus**2),qp/(1+a*minus**2)]
def fold_profile(z,t):
 def profile(k):return -np.expm1(-k*t)
 def err(k):
  q=profile(k); return np.linalg.norm(z-np.outer(z@q/(q@q),q))
 return profile(minimize_scalar(err,bounds=(.1,2),method='bounded').x)


def _new_b(panel, x, mode):
    u,v=panel['u'],panel['v']
    def phi(x,t):
        a,b=t
        if mode==0: return np.tanh(a*x**3)
        if mode==1: return np.tanh(a*x**3+b*x)
        if mode==2: return np.sin(a*x**3+b*x)
        if mode==3: return np.arcsinh(a*x**3+b*x)
        if mode==4: return np.tanh(a*x)+b*np.sin(1.7*x)
        if mode==5: return np.tanh(a*x**3)+b*np.tanh(1.7*x)
        if mode==6: return np.sin(a*x)+b*np.sin(1.7*x)
        if mode==7: return np.tanh(a*x)+b*np.tanh(1.7*x)
        if mode==8: return np.tanh(a*x)+b*x**3
        if mode==9: return np.tanh(a*x**3/(1+abs(b)*x*x))
        return np.tanh(a*x)+b*x**3/(1+x*x)
    def f(x,t): return np.c_[np.ones(len(x)),x,phi(x,t)]
    def r(t):
        F=f(u,t); H=np.linalg.pinv(F,rcond=1e-9)
        return ((F@(H@v)-v)/np.maximum(.04,1-(F*H.T).sum(1))[:,None]).ravel()
    fit=min((least_squares(r,t,bounds=([.05,-4],[8,4]),max_nfev=65) for t in ((.8,.15),(1.7,-.25),(3,.6))),key=lambda o:o.cost)
    th=fit.x; F=f(u,th)
    return f(x,th)@np.linalg.lstsq(F,v,rcond=1e-9)[0],np.mean(r(th)**2)**.5

def _scalar_features(b,kind,k):
    P=_polys(b,3); C=P[:,15:]; Q=P[:,5:15]
    if kind==0: return np.c_[P[:,:15],np.tanh(k*C)/k]
    if kind==1: return np.c_[P[:,:15],np.arcsinh(k*C)/k]
    if kind==2: return np.c_[P[:,:15],C/(1+k*C*C)]
    if kind==3: return np.c_[P[:,:5],np.tanh(k*Q)/k,np.tanh(k*C)/k]
    if kind==4: return np.c_[P[:,:15],np.tanh(k*C)/k,np.sin(k*Q)/k]
    if kind==5: return np.c_[P[:,:15],np.tanh(k*C)/k,np.tanh(k*_polys(b,4)[:,35:])/k]
    if kind==6: return np.c_[P[:,:15],np.tanh(k*C)/k,C]
    if kind==7: return np.c_[P[:,:5],Q,np.tanh(k*Q)/k,np.tanh(k*C)/k]
    if kind==8: return np.c_[P[:,:15],np.tanh(k*C)/k,_polys(np.tanh(k*b)/k,2)[:,1:]]
    i,j=np.triu_indices(4,1); S=(b[:,i]+b[:,j])**2; D=(b[:,i]-b[:,j])**2
    return np.c_[P[:,:15],np.tanh(k*C)/k,(Q[:,:,None]/(1+k*np.c_[S,D][:,None,:])).reshape(len(b),-1)]

def _scalar_fit(F,b,y,n,lam,scaling,link):
    A=F[:n]; scale=np.maximum(.25,A.std(0)) if scaling else 1.; G=(F-A.mean(0))/scale; T=G[:n]; mean=y.mean()
    if link in (1,2,-1,-2):
        c=max(abs(y))*({1:1.05,2:1.3,-1:1.15,-2:1.6}[link]); v=c*np.arctanh(y/c) if link>0 else c*np.tanh(y/c)
        mean=v.mean(); z=G[n:]@T.T@solve(T@T.T+lam*np.eye(n),v-mean,assume_a='pos')+mean
        return c*np.tanh(z/c) if link>0 else c*np.arctanh(np.clip(z/c,-.97,.97))
    if link==3:
        q=np.c_[b,b[:,np.arange(4)]**2,_polys(b,2)[:,5:]]; q=q/np.maximum(.3,q[:n].std(0)); X=np.c_[np.ones(n),T,-y[:,None]*q[:n]]
        reg=np.r_[1e-8,np.full(T.shape[1],lam),np.full(q.shape[1],lam*12+.02)]; X1=X/np.sqrt(reg); c=(X1.T@solve(X1@X1.T+np.eye(n),y,assume_a='pos'))/np.sqrt(reg)
        return (c[0]+G[n:]@c[1:1+T.shape[1]])/np.maximum(.4,1+q[n:]@c[1+T.shape[1]:])
    return G[n:]@T.T@solve(T@T.T+lam*np.eye(n),y-mean,assume_a='pos')+mean


def run_node(inputs: dict) -> dict:
    s=inputs.get("__input__",inputs); tr=s["train"]; n=len(tr["x"])
    w=_inverse(s["panel_a"],np.r_[tr["x"],s["test"]["x"]]); t=s["eval_times"]
    q=fold_profile(tr["traj"],t)
    Y=np.c_[tr["y"],tr["traj"]@q/(q@q)]
    c1=None
    preds=[]
    for col,(key,extra,theta,typ,a,lam,scaling) in enumerate([
        ('csat', 'lin', [1.7019903686123374, 1.0805054637640883], 'satcubic', 3.0, 0.02, True),
        ('csat', 'cube', [1.7149165936508188, 5.999999999999978], 'satcubic', 3.0, 0.08, True),
    ]):
        b=_calibrate_b(s["panel_b"],w,key,extra,theta)
        F=pf(b,typ,a,w,c1); A=F[:n]; center=A.mean(0); scale=np.maximum(.25,A.std(0)) if scaling else 1.
        A=(A-center)/scale; G=(F[n:]-center)/scale
        v=Y[:,col]; m=v.mean(); preds.append(G@A.T@solve(A@A.T+lam*np.eye(n),v-m,assume_a="pos")+m)
    b_scalar=_new_b(s["panel_b"],w,7)[0]
    Fs=_scalar_features(b_scalar,2,0.6)
    y=_scalar_fit(Fs,b_scalar,tr["y"],n,0.045,True,2); z=np.outer(preds[1],q)
    return {"y_hat":y,"state_pred":z,"diagnostics_vec":np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])}
