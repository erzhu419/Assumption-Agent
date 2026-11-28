"""Calibrated panel transfer and supervised scalar and trajectory readouts."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _panel(u, panel, key, kind, initial):
    x, y = panel['u'], panel[key]
    def design(t, x):
        return _aff(x + t[0]*kind(t[1]*x))
    def residual(t):
        a = design(t, x)
        return (a@_fit(a,y)-y).ravel()
    t = least_squares(residual, initial, max_nfev=120).x
    return design(t,u) @ _fit(design(t,x),y)


def _coords(st, z):
    b = st['panel_b']
    coef = _fit(_aff(b['u']),b['v'])
    u = np.linalg.solve(coef[1:].T, (z-coef[0]).T).T
    return _panel(u,b,'v',np.sin,[-.4,2.9])


def _quad(z):
    return np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(z.shape[1]),2)])


def _predict(st,z,zt):
    a,b=np.c_[_aff(z),_quad(z)],np.c_[_aff(zt),_quad(zt)]
    return b@_fit(a,np.column_stack((st['train']['y'],st['train']['traj'])))


def run_node(inputs: dict) -> dict:
    st, enc = inputs['__input__'], inputs['node_00']
    n = len(st['train']['x'])
    z = _coords(st,np.concatenate((enc['z'],enc['zt'])))
    p = _predict(st,z[:n],z[n:])
    y,t = p[:,0],p[:,1:]
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}

def _components(st, enc):
    from scipy.optimize import least_squares
    b=st['panel_b']; U=np.asarray(b['u']); V=np.asarray(b['v']); n=len(st['train']['x'])
    Z=np.r_[enc['z'],enc['zt']]; C=_fit(_aff(U),V); u=np.linalg.solve(C[1:].T,(Z-C[0]).T).T
    panels=[]
    for name, func in [('sin',lambda x,f: np.sin(f*x)),('tanh',lambda x,f:np.tanh(f*x)),('rat',lambda x,f:x/(1+f*x*x)),('cubic',lambda x,f:x**3),('power',lambda x,f:np.sign(x)*abs(x)**f),('sinh',lambda x,f:np.sinh(f*x)),('atan',lambda x,f:np.arctan(f*x)),('sin3',lambda x,f:np.sin(f*x)**3),('erf',lambda x,f:__import__('scipy').special.erf(f*x))]:
        def d(x,p):return _aff(x+p[0]*func(x,p[1]))
        def err(p):
            A=d(U,p);return (A@_fit(A,V)-V).ravel()
        fits=[least_squares(err,p,bounds=([-20,.12],[20,6]),max_nfev=140) for p in [(-.4,2.9),(-.9,1.2),(.8,2.1),(-1.5,3.6)]]
        o=min(fits,key=lambda o:o.cost); M=_fit(d(U,o.x),V); panels.append((np.std(o.fun),name,d(u,o.x)@M))
    for deg in (3,5,7):
        def d(x):return _aff(np.column_stack([x**j for j in range(1,deg+1,2)]))
        D=d(U); M=_fit(D,V); panels.append((np.std(D@M-V)+deg*.0002,'poly'+str(deg),d(u)@M))
    panels.sort(key=lambda p:p[0]); z=panels[0][2]; v=_panel(z,st['panel_c'],'v1',np.tanh,[2.,2.27]); w=_panel(z,st['panel_c'],'v2',np.cos,[-.65,1.29]); return u,z,v,w,panels


def _designs(st, enc):
    u,z,v,w,panels=_components(st,enc); groups=[]; n=len(st['train']['x']); N=len(z)
    def add(name,D):
        D=np.asarray(D).reshape(N,-1); D=D-D[:n].mean(0); D=D/np.maximum(D[:n].std(0),1e-8); groups.append((name,D))
    from itertools import combinations, combinations_with_replacement as cr
    for name,x in [('u',u),('z',z),('v',v),('w',w)]:
        add(name,x);add(name+'q',_quad(x));add(name+'c',np.column_stack([np.prod(x[:,jj],axis=1) for jj in cr(range(4),3)]));
    for name,x in [('u',u),('z',z)]:
        for f in [1.25,2.15,3.]:add(name+'trig'+str(f),np.c_[np.sin(f*x),np.cos(f*x),np.tanh(f*x)])
    for name,x in [('uz',np.c_[u,z]),('zv',np.c_[z,v]),('zw',np.c_[z,w]),('all',np.c_[u,z,v,w])]:
        add(name+'q',_quad(x));
    for name,x in [('u',u),('z',z),('uz',np.c_[u,z]),('zvw',np.c_[z,v,w])]:
        S=(x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-6); d=((S[:,None]-S[None,:])**2).mean(2)
        for h in [.45,1.,2.2,5.]:
            K=np.exp(-d/h**2/2); ev,E=np.linalg.eigh(K); add(name+'rbf'+str(h),E*np.sqrt(np.maximum(ev,0))) if False else groups.append((name+'rbf'+str(h),K))
        for h in [1.,2.2]:
            t=np.sqrt(3*d)/h;groups.append((name+'mat'+str(h),(1+t)*np.exp(-t)))
    kernels=[]
    for name,D in groups:
        K=D if 'rbf' in name or 'mat' in name else D@D.T/D.shape[1]; kernels.append(K)
    return panels,groups,np.array(kernels)


def _candidate(st, enc, report=False):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    n=len(st['train']['x']); T=np.asarray(st['train']['traj']); yy=np.asarray(st['train']['y']); Y=np.c_[yy,T];
    panels,groups,K=_designs(st,enc); N=K.shape[1]; names=[p[0] for p in groups];
    if report:print('panel',[(round(a,7),b) for a,b,_ in panels]);
    # Time-series modes are used for covariance estimation only, so prediction retains every output time.
    _,_,W=np.linalg.svd(T,full_matrices=False); R=np.c_[yy,T@W[0]]; R=(R-R.mean(0))/R.std(0); _,sv,vt=np.linalg.svd(R,full_matrices=False); R=R@vt.T; R/=np.maximum(sv/np.sqrt(n),.2)
    I=np.eye(n); A=np.concatenate([K[:,:n,:n],I[None]],axis=0); An=A.reshape(len(A),-1); one=np.ones(n); allmodels=[]
    def model(k,noise):
        H=k[:n,:n]+noise*I; L=cho_factor(H); iv=cho_solve(L,I); a=iv@one; P=iv-np.outer(a,a)/sum(a); beta=P@Y; mu=(a@Y)/sum(a); E=abs(beta/np.diag(P)[:,None]); p=k[n:,:n]@beta+mu; e=np.r_[E[:,0].mean(),E[:,1:].mean()]; return e,p
    def save(tag,k,noise):
        e,p=model(k,noise); allmodels.append((float(e.sum()),tag,e,p))
    for lam in [0.,.3,1.,3.]:
        def obj(t):
            w=np.exp(t); H=(w@An).reshape(n,n); L=cho_factor(H); iv=cho_solve(L,I); a=iv@one; P=iv-np.outer(a,a)/sum(a); al=P@R
            loss=.5*np.sum(R*al)+R.shape[1]*(np.log(np.diag(L[0])).sum()+.5*np.log(sum(a)))+lam*np.log1p(w[:-1]).sum(); B=.5*(R.shape[1]*P-al@al.T); g=w*(An@B.ravel()); g[:-1]+=lam*w[:-1]/(1+w[:-1]); return loss,g
        t=np.full(len(A),-4.);t[-1]=-6.; w=np.exp(minimize(obj,t,jac=True,method='L-BFGS-B',bounds=[(-13,6)]*(len(A)-1)+[(-11,1)],options={'maxiter':220}).x); save('ML'+str(lam),np.tensordot(w[:-1],K,axes=1),w[-1]);
    for j,(name,D) in enumerate(groups):
        if name in ('u','z','v','w'):continue
        for baseline in [('u','z'),('z',),('u','z','v','w')]:
            B=sum(K[names.index(b)] for b in baseline)*2
            for noise in (.0001,.001,.008,.04):save((name,baseline,noise),K[j]+B,noise)
    allmodels.sort(key=lambda q:q[0]);
    if report:print('internal',[(v[1],np.round(v[2],5)) for v in allmodels[:8]],'ML',[(v[1],np.round(v[2],5)) for v in allmodels if str(v[1]).startswith('ML')]);
    return allmodels
