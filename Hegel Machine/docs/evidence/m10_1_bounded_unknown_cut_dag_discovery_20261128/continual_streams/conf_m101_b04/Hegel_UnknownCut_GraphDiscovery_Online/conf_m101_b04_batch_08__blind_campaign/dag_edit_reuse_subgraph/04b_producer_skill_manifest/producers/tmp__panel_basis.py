import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; c=s['panel_c']; u=c['u']; N=np.arange(len(x)); f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); af=f['_affine']; a=af(x)@f['_panel'](s['panel_a'],3); w=af(a)@f['_panel'](s['panel_b']); Y=c['v1']
def basis(X,freq,mode):
    args=X*freq
    if mode==0: return np.c_[np.ones(len(X)), np.sin(args),np.cos(args)]
    if mode==1: return np.c_[np.ones(len(X)), X, np.sin(args),np.cos(args)]
    return np.c_[np.ones(len(X)), X, np.sin(args)]
def fit(freq,mode):
    F=basis(u,freq,mode); return np.linalg.lstsq(F,Y,rcond=None)[0]
models=[]
for mode in range(3):
    def residual(freq): return (basis(u,freq,mode)@fit(freq,mode)-Y).ravel()
    opt=min((least_squares(residual,np.full(4,k),bounds=(.15,8),max_nfev=150) for k in (1.5,2.5,3.5,4.5,5.5)),key=lambda q:np.linalg.norm(q.fun)); freq=opt.x; co=fit(freq,mode); print('panel',mode,freq.round(4),np.std(opt.fun)); models.append((mode,freq,co))
vs=[('official',N[52:]),('radius',np.argsort(np.linalg.norm(x,axis=1))[-12:])]+[(str(i),v) for i,v in enumerate(np.array_split(np.random.default_rng(427).permutation(len(x)),4))]+[(f'{i}:{d}',np.argsort(d*x[:,i])[-12:]) for i in range(4) for d in (-1,1)]
def trend(X):
    D=X@X.T; return 1+D+(D**2+(X*X)@(X*X).T)/2
def radial(X,k):return np.exp(-k*np.sum((X[:,None]-X[None,:])**2,axis=-1))
def evaluate(K,r):
    e=[]
    for _,v in vs:
     t=np.setdiff1d(N,v); p=K[np.ix_(v,t)]@np.linalg.solve(K[np.ix_(t,t)]+r*np.eye(len(t)),y[t]); e.append(np.mean(abs(p-y[v])))
    return np.array(e)
Q=trend(x); old=evaluate(Q+radial(x,.5),.001); print('installed',old.round(4)); print('calibrated',evaluate(Q+radial(w,1),.002).round(4)); results=[]
for coord,X in [('x',x),('a',a),('w',w)]:
 for mode,ff,co in models:
    Z=basis(X,ff,mode)@co; S=trend(Z); J=Z@Z.T
    for weight in (.1,1,10):
     for tag,K in [('replace',S),('combine',Q+weight*S),('rbf',Q+weight*radial(Z,1)),('interaction',Q+weight*J*(1+x@x.T)),('additive',Q+weight*(basis(X,ff,mode)@basis(X,ff,mode).T))]:
      for mix in (0,.3,1.):
       r=.002; e=evaluate(K+mix*radial(w,1),r); results.append(((coord,mode,weight,tag,mix),e))
for lab,e in sorted((r for r in results if r[1][0]<=.038),key=lambda r:r[1].max())[:10]: print(lab,e.round(4))
