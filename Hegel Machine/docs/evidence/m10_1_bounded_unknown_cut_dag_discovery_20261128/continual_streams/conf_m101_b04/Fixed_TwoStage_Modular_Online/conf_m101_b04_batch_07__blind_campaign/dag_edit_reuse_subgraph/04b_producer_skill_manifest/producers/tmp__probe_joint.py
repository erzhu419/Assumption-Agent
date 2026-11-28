import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})
z,q,y,T=a['z_tr'],a['z_te'],a['y'],a['traj']; u,v=(a['panel_b'][k] for k in ('u','v')); t=a['eval_times']
from scipy.optimize import least_squares
from itertools import combinations_with_replacement

def fit(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def basis(x,f): return np.column_stack((np.ones(len(x)),f))
def form(k,x,kind):
 s=x*k[0]
 fs={'sin':np.sin,'tanh':np.tanh,'rat':lambda s:s/(1+s*s),'atan':np.arctan}
 return fs[kind](s)+k[1]*x

def panel(kind,shared):
 def des(k,x): return basis(x,form(k,x,kind) if shared else np.column_stack((x,form([k[0],0],x,kind))))
 def resid(k):
  U=des(k,u); return (U@fit(U,v)-v).ravel()
from itertools import combinations
from scipy.optimize import minimize_scalar
opt=least_squares(lambda k: ( (D:=basis(u,np.sin(k[0]*u)+k[1]*u))@fit(D,v)-v).ravel(),[2.86,-2.52]); k=opt.x
D=lambda x: basis(x,np.sin(k[0]*x)+k[1]*x)
G=D(z)@fit(D(u),v)
print('t',t,'panel',k,'residual',np.sqrt(np.mean(opt.fun**2)))
from scipy.optimize import least_squares

def readout(g,z,y,h,j,kind):
    B=np.column_stack((np.ones(len(g)),g,h)); Q=np.linalg.qr(B[j])[0]
    def f(p):
        x=g@p[:4]+p[4]
        return np.sin(x) if kind=='sin' else np.tanh(x)
    r=y[j]-Q@(Q.T@y[j])
    def err(p):
        a=f(p)[j]; a=a-Q@(Q.T@a); return a*(a@r/max(a@a,1e-14))-r
    starts=np.random.default_rng(2).normal(size=(12,5)); opts=[least_squares(err,p,max_nfev=200) for p in starts]; o=min(opts,key=lambda p:np.linalg.norm(p.fun)); F=np.c_[B,f(o.x)]; w=fit(F[j],y[j]); return F@w,np.sqrt(np.mean(o.fun**2)),o.x,w
n=len(z); rng=np.random.default_rng(17); folds=list(np.array_split(rng.permutation(n),5))+[np.arange(52,n),np.argsort(np.max(abs(z),axis=1))[-15:]]
H={'none':np.empty((n,0)), 'z0sq':z[:,[0]]**2,'zsq':z**2,'z0quartic':np.c_[z[:,0]**2,z[:,0]**4],'z0shiftcube':np.c_[z[:,0]**2,z[:,0]**3],'G13':(G[:,1]*G[:,3])[:,None],'G013':np.prod(G[:,[0,1,3]],axis=1)[:,None]}
for k,h in H.items():
 for kind in ('sin','tanh'):
    vals=[]
    for ix in folds:
        j=np.setdiff1d(np.arange(n),ix); p,rm,w,b=readout(G,z,y,h,j,kind); vals.append(np.mean(abs(p[ix]-y[ix])))
    p,rm,w,b=readout(G,z,y,h,np.arange(n),kind)
    print(k,kind,'CV',np.round(vals,5),'fit',round(rm,5),'index',np.round(w,3),'coefs',np.round(b,3))
F=np.c_[np.ones(n),z,G]
for h in folds:
 j=np.setdiff1d(np.arange(n),h); P=F[h]@fit(F[j],T[j]); print('traj',np.round(np.mean(abs(P-T[h]),axis=0),5))
