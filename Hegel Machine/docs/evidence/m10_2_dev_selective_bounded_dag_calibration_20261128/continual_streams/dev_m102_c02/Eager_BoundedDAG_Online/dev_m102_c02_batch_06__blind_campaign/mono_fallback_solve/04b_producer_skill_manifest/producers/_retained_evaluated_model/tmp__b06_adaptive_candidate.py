"""Recover latent measurements and learn regularized nonlinear readouts."""
import numpy as np
from scipy.optimize import root, minimize_scalar, least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _latent(study, raw):
    p = study['panel_a']; u,v = p['u'],p['v']
    ok = np.max(abs(u-v),axis=1)<2; u,v = u[ok],v[ok]
    h = study['telemetry_vec'][1]; w=np.roll(u,-1,axis=1)
    b=np.column_stack([_ls(_aff(np.c_[u[:,j],w[:,j],u[:,j]*w[:,j],np.sin(h*u[:,j])]),v[:,j]) for j in range(4)])
    def forward(q):
        r=np.roll(q,-1)
        return b[0]+b[1]*q+b[2]*r+b[3]*q*r+b[4]*np.sin(h*q)
    out=[]
    for row in raw:
        starts=[row,np.zeros(4),row*.7,row*1.3,*u[np.argsort(np.linalg.norm(v-row,axis=1))[:6]]]
        roots=[root(lambda q:forward(q)-row,start).x for start in starts]
        out.append(min(roots,key=lambda q:np.linalg.norm(q) if np.linalg.norm(forward(q)-row)<1e-7 else 1e3+np.linalg.norm(forward(q)-row)))
    return np.asarray(out)


def _coordinates(study, raw):
    x=_latent(study,raw); ops=[]
    for panel,key,fun,br in [('panel_b','v',lambda z,a:np.tanh(a*z),(.5,2.5)),('panel_c','v1',lambda z,a:z/(1+a*abs(z)),(.2,2.0)),('panel_c','v2',lambda z,a:np.cos(a*z),(.5,2.2))]:
        p=study[panel]; u,v=p['u'],p[key]
        f=lambda z,a:_aff(np.c_[z,fun(z,a)])
        def obj(a):
            m=f(u,a); return np.linalg.norm(v-m@_ls(m,v))
        a=minimize_scalar(obj,bounds=br,method='bounded').x; b=_ls(f(u,a),v)
        ops.append(lambda z,f=f,a=a,b=b:f(z,a)@b)
    b,c,d=(op(x) for op in ops)
    return {'x':x,'b':b,'c':c,'d':d,'bc':ops[1](b),'bd':ops[2](b),'cb':ops[0](c),'cd':ops[2](c),'db':ops[0](d),'dc':ops[1](d)}


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


def solve_batch(study:dict)->dict:
    train=study['train']; raw=np.r_[train['x'],study['test']['x']]; D=_coordinates(study,raw)
    times=study['eval_times']; T=train['traj']; Y=np.c_[train['y'],T]
    k,h=study['telemetry_vec'][:2]; out=_adaptive(D,Y,k,h)
    def basis(r): return np.array([times,-np.expm1(-r[0]*times),-np.expm1(-r[1]*times)])
    def res(r):
        b=basis(r); return (T-_ls(b.T,T.T).T@b).ravel()
    r=least_squares(res,[.8,7],bounds=([.1,2],[2,20])).x; b=basis(r)
    y=out[:,0].copy(); p=_ls(b.T,out[:,1:].T).T@b
    return {'y_hat':y,'state_pred':p,'diagnostics_vec':np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])}
