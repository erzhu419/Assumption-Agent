import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda A,b:np.linalg.lstsq(A,b,rcond=1e-11)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
i,j=np.triu_indices(4); ia,ja=np.triu_indices(4,1)
quad=lambda x:np.c_[aff(x),x[:,i]*x[:,j]]
cal=lambda x:np.c_[quad(x),x**3]
u,v=s['panel_a']['u'],s['panel_a']['v']; F=cal(u); k=np.arange(len(u))
for _ in range(14):
 c=ls(F[k],v[k]); r=np.linalg.norm(F@c-v,axis=1); k=np.argsort(r)[:40]
k=r<max(.001,6*np.median(r)); c=ls(F[k],v[k]); print('calibration',k.sum(),np.round(c,4))
def inv(x):
 z=(x-c[0])@np.linalg.inv(c[1:5]);
 for _ in range(30):
  r=cal(z)@c-x; D=np.stack([(cal(z+e*1e-5)@c-cal(z-e*1e-5)@c)/2e-5 for e in np.eye(4)],axis=2); z-=np.clip(np.linalg.solve(D,r[...,None])[...,0],-.45,.45)
 return z
x=inv(s['train']['x']); print('latent range',x.min(0),x.max(0),'roundtrip',np.max(abs(cal(x)@c-s['train']['x'])))
forms={'sin':lambda z,p:np.sin(p[0]*z),'cos':lambda z,p:np.cos(p[0]*z),'tanh':lambda z,p:np.tanh(p[0]*z),'exp':lambda z,p:z*np.exp(-p[0]*z*z),'rational':lambda z,p:z/(1+p[0]*z*z),'sinh':lambda z,p:np.sinh(p[0]*z),'expp':lambda z,p:z*np.exp(-p[0]*abs(z)**p[1]),'evenexp':lambda z,p:np.exp(-p[0]*z*z),'cosplus':lambda z,p:np.cos(p[0]*z)+p[1]*np.sin(p[0]*z),'power':lambda z,p:z*abs(z)**p[0]}
ops={}
for name,panel,key in [('b','panel_b','v'),('c','panel_c','v1'),('d','panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][key]; fits=[]
 for kind,g in forms.items():
  for base in ['one','aff','affsquare','cross','quad']:
   def f(z,p,g=g,base=base):
    a={'one':np.ones((len(z),1)),'aff':aff(z),'affsquare':np.c_[aff(z),z*z],'cross':np.c_[aff(z),z[:,ia]*z[:,ja]],'quad':quad(z)}[base]; return np.c_[a,g(z,p)]
   def res(p):
    A=f(u,p); return (A@ls(A,v)-v).ravel()
   bounds=([.03,.15],[8,7]) if kind=='expp' else (([.03,-8],[8,8]) if kind=='cosplus' else ([.03],[8]))
   opts=[least_squares(res,[t,2] if len(bounds[0])==2 else [t],bounds=bounds,max_nfev=100) for t in [.4,1.3,2.7,4.5]]; p=min(opts,key=lambda z:z.cost).x; A=f(u,p); w=ls(A,v); err=A@w-v; hat=np.einsum('ij,ji->i',A,np.linalg.pinv(A)); loo=np.mean(abs(err)/np.maximum(1-hat[:,None],1e-8)); fits.append((loo,np.mean(abs(err)),kind,base,p,lambda z,f=f,p=p,w=w:f(z,p)@w))
 fits.sort(key=lambda z:z[0]); print(name,[(round(l,5),round(e,5),k,b,np.round(p,4)) for l,e,k,b,p,_ in fits[:10]]); ops[name]=[f[-1] for f in fits[:3]]
y=np.c_[s['train']['y'],s['train']['traj']]; features={'x':x}
for name,of in ops.items():
 for oi,o in enumerate(of):
  features[name+str(oi)]=o(x)
  for other,pf in ops.items():
   features[other+name+str(oi)]=pf[0](o(x))
for name in ['b','c','d']:
 for other in ['b','c','d']:
  if name<other: features[name+'+'+other]=np.c_[features[name+'0'],features[other+'0']]
allr=[]
for name,z in features.items():
 for degree in (1,2,3):
  if z.shape[1]>4 and degree>1: continue
  F=np.column_stack([np.prod(z[:,ix],axis=1) for k in range(degree+1) for ix in it.combinations_with_replacement(range(z.shape[1]),k)]); pred=F@ls(F[:52],y[:52]); err=abs(pred-y); allr.append((err[52:,0].mean(),err[52:,1:].mean(),err[:52,0].mean(),err[:52,1:].mean(),name,degree))
for col in (0,1): print('ranked errors',col,sorted(allr,key=lambda q:q[col])[:14])
