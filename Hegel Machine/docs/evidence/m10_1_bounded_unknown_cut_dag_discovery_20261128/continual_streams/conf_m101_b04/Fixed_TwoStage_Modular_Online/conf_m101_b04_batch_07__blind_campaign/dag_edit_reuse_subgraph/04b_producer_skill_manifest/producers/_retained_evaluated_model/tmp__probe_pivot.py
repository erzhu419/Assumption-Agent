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

from itertools import combinations
n=len(z); rng=np.random.default_rng(17); folds=list(np.array_split(rng.permutation(n),5))+[np.arange(52,n),np.argsort(np.max(abs(z),axis=1))[-15:]]
starts=np.random.default_rng(2).normal(size=(12,5))

def check(X,B,typ,j,h):
    A=B[j]; Q=np.linalg.qr(A)[0]; rd=lambda x:x-Q@(Q.T@x); r=rd(y[j]); Xj=X[j]
    def f(x,p):
        w=x@p[:4]+p[4]; return np.sin(w) if typ!='tanh' else np.tanh(w)
    def nonlinear(x,p):
        if typ=='double': return np.column_stack((f(x,p[:5]),f(x,p[5:])))
        if typ=='mod': return np.column_stack((f(x,p), f(x,p)*x[:,0]))
        return f(x,p)[:,None]
    def res(p):
        V=rd(nonlinear(Xj,p)); return V@fit(V,r)-r
    seeds=starts if typ!='double' else np.column_stack((starts,np.roll(starts,3,axis=0)))
    os=[least_squares(res,p,max_nfev=140) for p in seeds]
    op=min(os,key=lambda o:np.linalg.norm(o.fun)); F=np.c_[B,nonlinear(X,op.x)]; w=fit(F[j],y[j]); return F@w,op.x,w

B=np.c_[np.ones(n),G]; cross=np.prod(G[:,[0,1,3]],axis=1)
models=[('base',G,B,'sin'),('trip',G,np.c_[B,cross],'sin'),('rawindex',z,B,'sin'),('double',G,B,'double'),('mod',G,B,'mod')]
for label,extra in [('trip+pair',np.c_[cross,G[:,1]*G[:,3]]),('trip+z0^2',np.c_[cross,z[:,0]**2]),('trip+z0^4',np.c_[cross,z[:,0]**4]),('trip+all3',np.column_stack([np.prod(G[:,ij],axis=1) for ij in combinations(range(4),3)]))]:
    models.append((label,G,np.c_[B,extra],'sin'))
for label,X,A,typ in models:
    errs=[]
    for h in folds:
        j=np.setdiff1d(np.arange(n),h); pp,*_=check(X,A,typ,j,h); errs.append(np.mean(abs(pp[h]-y[h])))
    pp,w,b=check(X,A,typ,np.arange(n),[]); e=pp-y
    print(label,'CV',np.round(errs,5),'RMSE',round(np.std(e),5),'p',np.round(w,3),'w',np.round(b,3),flush=True)
    if label in ('base','trip','double'):
        ix=np.argsort(abs(e))[-5:]; print('worst z, G, y, e:',np.round(np.c_[z[ix],G[ix],y[ix],e[ix]],3),flush=True)
