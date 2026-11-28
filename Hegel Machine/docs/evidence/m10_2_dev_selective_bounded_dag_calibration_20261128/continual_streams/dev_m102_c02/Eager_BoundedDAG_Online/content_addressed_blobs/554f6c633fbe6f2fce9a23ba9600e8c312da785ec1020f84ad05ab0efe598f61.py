"""Recover latent measurements and learn regularized nonlinear readouts."""
import numpy as np
from scipy.optimize import root, minimize_scalar, least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.linear_model import lasso_path


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


def _bank(D, k, h, n):
    a=[]; names=[]
    def add(z,name):
        z=np.asarray(z); z=z[:,None] if z.ndim==1 else z
        a.append(z); names.extend(f'{name}:{j}' for j in range(z.shape[1]))
    for name,z in D.items():
        add(z,name); add(z*z,name+'^2'); add(z**3,name+'^3')
        add(np.sin(h*z),name+' sin'); add(np.cos(h*z),name+' cos'); add(np.tanh(h*z),name+' tanh'); add(abs(z),name+' abs'); add(z/(1+k*z*z),name+' rat'); add(z/(1+.85*abs(z)),name+' soft')
        for i in range(4):
            for j in range(i+1,4):
                p,q=z[:,i],z[:,j]; tag=f'{name}{i}{j}'
                add(p*q,tag+' product'); add((p*q)/(1+k*(p-q)**2),tag+' rational product'); add((p*q)/(1+k*(p+q)**2),tag+' rational product sum'); add(abs(p-q),tag+' absdiff'); add(abs(p+q),tag+' abssum')
                add(np.sin(h*(p-q)),tag+' sindiff'); add(np.sin(h*(p+q)),tag+' sinsum'); add(np.cos(h*(p-q)),tag+' cosdiff'); add(np.cos(h*(p+q)),tag+' cossum')
        add(z/(1+k*np.sum(z*z,axis=1)[:,None]),name+' radial')
    for ni,nj in [('x','b'),('x','c'),('x','d'),('b','c'),('b','d'),('c','d'),('bc','bd')]:
        p,q=D[ni],D[nj]
        add((p[:,:,None]*q[:,None,:]).reshape(len(p),-1),ni+'*'+nj)
        add((p[:,:,None]/(1+k*q[:,None,:]**2)).reshape(len(p),-1),ni+'/'+nj)
    z=np.column_stack(a); sd=z[:n].std(0); ok=sd>1e-7
    return (z[:,ok]-z[:n,ok].mean(0))/sd[ok],np.asarray(names)[ok]


def _fit(D,Y,k,h,debug=False):
    n,m=Y.shape; N=len(D['x']); folds=np.array_split(np.random.default_rng(913).permutation(n),5)
    A,names=_bank(D,k,h,n); K={}
    for name,Z in D.items():
        Z=(Z-Z[:n].mean(0))/np.maximum(Z[:n].std(0),.0001)
        dot=Z@Z.T/4; dist=cdist(Z,Z,'sqeuclidean')
        K[name+' p2']=(1+dot)**2; K[name+' p3']=(1+dot)**3
        for w in (2.5,5,9): K[f'{name} g{w}']=np.exp(-dist/(2*w*w))
    for subset in [('x','b'),('x','b','c','d'),('x','b','bc','bd')]:
        for suffix in [' p2',' p3',' g2.5',' g5',' g9']:
            K[str(subset)+suffix]=sum(K[s+suffix] for s in subset)/len(subset)
    K['dictionary']=1+A@A.T/A.shape[1]
    settings=[(name,lam) for name in K for lam in (1e-5,.001,.01,.06)]
    def ridge(name,lam,i,j):
        M=K[name]; center=Y[i].mean(0)
        return M[np.ix_(j,i)]@solve(M[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i]-center,assume_a='pos')+center
    pred=[]; errs=[]; tags=[]; it=np.arange(n); jt=np.arange(n,N)
    for name,lam in settings:
        P=np.empty_like(Y)
        for j in folds:
            i=np.setdiff1d(it,j); P[j]=ridge(name,lam,i,j)
        errs.append(np.mean(abs(P-Y),0)); tags.append((name,lam)); pred.append(ridge(name,lam,it,jt))
    ratios=np.array([.12,.07,.04,.025,.015,.009,.005,.0025])
    def sparse(i,j):
        u=A[i]; mu=u.mean(0); Z=u-mu; means=Y[i].mean(0); y=Y[i]-means
        aout=np.empty((len(ratios),len(j),m)); coef=[]
        for t in range(m):
            top=max(np.max(abs(Z.T@y[:,t]))/len(i),1e-8)
            _,b,_=lasso_path(Z,y[:,t],alphas=ratios*top,max_iter=3500,tol=.00025)
            aout[:,:,t]=((A[j]-mu)@b).T+means[t]; coef.append(b)
        return aout,coef
    P=np.empty((len(ratios),n,m))
    for j in folds:
        P[:,j],_=sparse(np.setdiff1d(it,j),j)
    Q,co=sparse(it,jt)
    for a in range(len(ratios)):
        errs.append(np.mean(abs(P[a]-Y),0)); tags.append(('lasso',ratios[a])); pred.append(Q[a])
    er=np.asarray(errs); pp=np.asarray(pred); out=np.empty((len(jt),m))
    for group in [np.array([0]),np.arange(1,m)]:
        loss=er[:,group].mean(1); ids=np.argsort(loss)[:6]; w=np.exp(-20*(loss[ids]/loss[ids[0]]-1)); out[:,group]=np.einsum('i,ijk->jk',w/w.sum(),pp[ids][:,:,group])
        if debug: print('CV candidates',[(tags[i],round(loss[i],5)) for i in ids]); print('lasso features',[(names[j],round(co[group[0]][j,4],3)) for j in np.argsort(abs(co[group[0]][:,4]))[-15:][::-1]])
    return out


def solve_batch(study:dict)->dict:
    train=study['train']; raw=np.r_[train['x'],study['test']['x']]; D=_coordinates(study,raw)
    times=study['eval_times']; T=train['traj']; Y=np.c_[train['y'],T]
    k,h=study['telemetry_vec'][:2]; out=_fit(D,Y,k,h)
    def basis(r): return np.array([times,-np.expm1(-r[0]*times),-np.expm1(-r[1]*times)])
    def res(r):
        b=basis(r); return (T-_ls(b.T,T.T).T@b).ravel()
    r=least_squares(res,[.8,7],bounds=([.1,2],[2,20])).x; b=basis(r)
    y=out[:,0].copy(); p=_ls(b.T,out[:,1:].T).T@b
    return {'y_hat':y,'state_pred':p,'diagnostics_vec':np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])}
