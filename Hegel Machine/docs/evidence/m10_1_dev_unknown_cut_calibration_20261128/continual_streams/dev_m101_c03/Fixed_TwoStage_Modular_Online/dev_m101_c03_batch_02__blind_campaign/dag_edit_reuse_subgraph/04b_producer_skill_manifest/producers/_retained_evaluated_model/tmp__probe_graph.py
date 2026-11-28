import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
aug=lambda x:np.column_stack((np.ones(len(x)),x))
a,b,c=(s['panel_'+k] for k in 'abc')
u=c['u']; ones=np.ones_like(u); v=c['v1']
opts=[]
for name,f in [('atan',np.arctan),('tanh',np.tanh),('softsign',lambda x:x/(1+abs(x))),('sqrt',lambda x:x/np.sqrt(1+x*x)),('erf',__import__('scipy.special',fromlist=['erf']).erf),('sin',np.sin),('rational',lambda x:x/(1+x*x)),('asinh',np.arcsinh)]:
 for lin,shift,ind in product((False,True),repeat=3):
  n=4 if ind else 1
  def model(x,p):
   w=f(x*p[:n]+(p[n:] if shift else 0)); return aug(np.c_[x,w] if lin else w)
  def res(p):
   X=model(u,p);return (X@ls(X,v)-v).ravel()
  p=least_squares(res,np.r_[np.full(n,1.7),np.zeros(n*shift)],bounds=(np.r_[np.full(n,.15),np.full(n*shift,-3.)],np.r_[np.full(n,5.),np.full(n*shift,3.)]),max_nfev=200).x
  X=model(u,p); E=res(p).reshape(v.shape); loo=E/(1-np.sum(X*np.linalg.pinv(X).T,axis=1))[:,None]
  opts.append((np.mean(E**2)**.5,np.mean(loo**2)**.5,name,lin,shift,ind,p,ls(X,v)))
for opt in sorted(opts,key=lambda o:o[0])[:18]:print('V1',opt[:-1])
C=ls(aug(a['u']),a['v']); X=aug(a['u']); y=a['v']
for _ in range(20):
 r=np.linalg.norm(y-X@C,axis=1); w=1/np.sqrt(1+(r/max(.0004,.8*np.median(r)))**2); C=ls(X*w[:,None],y*w[:,None])
r=np.linalg.norm(y-X@C,axis=1); print('A residuals',np.sort(r)[[0,20,44,45,46,47]])
B=ls(aug(b['u']),b['v']); V2=ls(aug(np.cos(2.4*u)),c['v2'])
from scipy.special import erf
fs={'atan':np.arctan,'tanh':np.tanh,'sqrt':lambda x:x/np.sqrt(1+x*x),'erf':erf,'sin':np.sin,'asinh':np.arcsinh,'softsign':lambda x:x/(1+abs(x)),'rational':lambda x:x/(1+x*x)}
fit=sorted(opts,key=lambda o:o[0])[0]; _,_,name,lin,shift,ind,p,V1=fit; n=4 if ind else 1
print('V1 coefficients',V1)
def cv1(x):
 w=fs[name](x*p[:n]+(p[n:] if shift else 0));return aug(np.c_[x,w] if lin else w)@V1
x=s['train']['x']; z=(x-C[0])@np.linalg.inv(C[1:]); zf=aug(x)@C
sources={'x':x,'a-':z,'a+':zf}
for k,w in list(sources.items()):
 sources[k+'b+']=aug(w)@B; sources[k+'b-']=(w-B[0])@np.linalg.inv(B[1:])
t=s['eval_times'][:,None]; traj=s['train']['traj']; y=s['train']['y']
def basis(r):return -np.expm1(-t*r)
def fun(r):
 T=basis(r);return (T@ls(T,traj.T)-traj.T).ravel()
r=least_squares(fun,[1.8,.4],bounds=([1.,.01],[10.,1.])).x; T=basis(r); amp=ls(T,traj.T).T; print('rates',r,'temporal RMSE',np.mean(fun(r)**2)**.5)
target=np.c_[y,amp[:,0]]
def pred(X,y,idx,j):
 H=X[idx]; p=np.eye(H.shape[1])*.00015;p[0,0]=0;return X[j]@np.linalg.solve(H.T@H+p,H.T@y[idx])
i=np.arange(52);j=np.arange(52,72)
L={k:aug(w) for k,w in sources.items()}; L.update({k+' + '+l:aug(np.c_[w,v]) for k,w in sources.items() for l,v in sources.items() if k<l}); L.update({k+'-v1':aug(cv1(w)) for k,w in sources.items()});L.update({k+'-v2':aug(aug(np.cos(2.4*w))@V2) for k,w in sources.items()});L['plain']=np.ones((len(x),1)); L={k:w for k,w in L.items() if not (w.shape[1]>5 and np.linalg.matrix_rank(w)<w.shape[1])}
D={k+'v1':cv1(w) for k,w in sources.items()}; E={k+'v2':aug(np.cos(2.4*w))@V2 for k,w in sources.items()}; D.update(sources); E.update(sources)
R=[]
for (k,d),(l,e) in product(D.items(),E.items()):
 dot=(d*e).sum(1); slow=d[:,0]*e[:,1]
 for m,h in L.items():
  X=np.c_[h,dot]; Q=np.c_[h,slow]; M=pred(X,target,i,j); S=pred(Q,amp[:,1],i,j); err=np.mean(abs(M[:,0]-y[j])); et=np.mean(abs(np.c_[M[:,1],S]@T.T-traj[j]));R.append((max(err,et),err,et,k,l,m))
for g in sorted(R)[:20]:print('GRAPH',g)
