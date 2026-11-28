exec(open('/tmp/operators.py').read().split('ops={}')[0])
from scipy.special import erf
p=s['panel_b']; u,v=p['u'],p['v']; rng=np.random.default_rng(41); folds=np.array_split(rng.permutation(len(u)),6)
def fun(z,name,w):
 if name=='tanh':return np.tanh(w*z)
 if name=='rat':return z/(1+(w*z)**2)
 if name=='erf':return erf(w*z)
 if name=='sin':return np.sin(w*z)
 if name=='atan':return np.arctan(w*z)
 if name=='mix':return np.column_stack((np.tanh(w[0]*z),np.sin(w[1]*z)))
def d(z,name,w,aff):return np.c_[lin(z) if aff else np.ones(len(z)),fun(z,name,w)]
def model(u,v,name,aff,init):
 def resid(q):
  a=d(u,name,q if name=='mix' else q[0],aff);return (a@fit(a,v)-v).ravel()
 o=least_squares(resid,init,bounds=(.25,4),max_nfev=100); q=o.x if name=='mix' else o.x[0]; b=fit(d(u,name,q,aff),v); return lambda z:d(z,name,q,aff)@b,q,np.mean(o.fun**2)**.5
rng=np.random.default_rng(710); ids=[np.arange(72),*[rng.permutation(72) for _ in range(8)],*[np.argsort(abs(z[:,j])) for j in range(4)]]
def cv(f,weight=0):
 res=[]
 for k,ix in enumerate(ids):
  n=56 if 0<k<9 else 52; i,j=ix[:n],ix[n:]; q=f[i]; scale=np.maximum(q.std(0),.03); q=q/scale
  c=fit(q,Y[i]) if not weight else np.linalg.solve(q.T@q+weight*np.eye(q.shape[1]),q.T@Y[i]); e=abs(f[j]/scale@c-Y[j]); res.append([e[:,0].mean(),e[:,1:].mean()])
 return np.round(np.r_[res[0],np.mean(res[1:9],axis=0),np.max(res[9:],axis=0)],5)
for name,init in [('tanh',[1.4]),('rat',[.9]),('erf',[1.2]),('atan',[1.3]),('sin',[2.8]),('mix',[1.4,2.6])]:
 for aff in (False,True):
  pred,q,e=model(u,v,name,aff,init); err=[]
  for j in folds:
   i=np.setdiff1d(np.arange(len(u)),j); h,_,_=model(u[i],v[i],name,aff,init);err.extend((h(u[j])-v[j]).ravel())
  b=pred(z); F=poly(b,2); print(name,aff,q,round(e,6),round(np.mean(np.square(err))**.5,6),cv(F),cv(np.c_[F,z],.001),cv(np.c_[F,z,b**3],.001),flush=True)
