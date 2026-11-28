import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement as cw
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x); ix=np.arange(n); np.set_printoptions(precision=5,suppress=True)
def aff(u):return np.c_[np.ones(len(u)),u]
def ls(f,v):return np.linalg.lstsq(f,v,rcond=None)[0]
def panel(p,k,func,guess):
 u,v=s[p]['u'],s[p][k]; idx=np.arange(len(u));
 if p=='panel_a':idx=np.where(np.max(np.abs(v),1)<3)[0]
 def fit(z):
  F=func(u[idx],z); c=ls(F,v[idx]);return F,c,v[idx]-F@c
 r=least_squares(lambda z:fit(z)[2].ravel(),guess,bounds=(.05,5.));F,c,e=fit(r.x);print(p,k,r.x, 'rmse',np.sqrt(np.mean(e**2)), 'loo',np.sqrt(np.mean((e/(1-np.diag(F@np.linalg.pinv(F)))[:,None])**2)));return lambda q:func(q,r.x)@c,c
A=s['panel_a']; u,v=A['u'],A['v']; ids=np.where(np.max(np.abs(v),1)<3)[0]
for j in range(4):
 k=j^2; z=u[ids,j]; w=u[ids,k]; tar=v[ids,j];
 def fun(t):return t[0]+t[1]*z+t[2]*np.sin(t[3]*w)+t[4]*np.sin(t[5]*z)-tar
 r=least_squares(fun,[0,1,.3,2.15,.1,1.25],max_nfev=1000);print('A sparse',j,r.x,np.sqrt(np.mean(r.fun**2)))
forms={'sin':lambda u,z:np.c_[aff(u),np.sin(z[0]*u)],'tanh':lambda u,z:np.c_[aff(u),np.tanh(z[0]*u)],'sin_cos':lambda u,z:np.c_[aff(u),np.sin(z[0]*u),np.cos(z[0]*u)],'tanh_only':lambda u,z:aff(np.tanh(z[0]*u)),'rational':lambda u,z:aff(u/np.sqrt(1+z[0]*u*u)),'sin_only':lambda u,z:aff(np.sin(z[0]*u)),'aff':lambda u,z:aff(u)}
for p,k in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 for name,fn in forms.items(): print(name,end=' '); panel(p,k,fn,[1.6])
af,ac=panel('panel_a','v',forms['sin'],[2.15]); u0=(x-ac[0])/ac[1:5].diagonal(); z=least_squares(lambda z:(af(z.reshape(x.shape))-x).ravel(),u0.ravel(),max_nfev=30).x.reshape(x.shape)
B=s['panel_b']; bm=ls(aff(B['u']),B['v']); c1,_=panel('panel_c','v1',forms['tanh_only'],[1.25]); c2,_=panel('panel_c','v2',forms['sin_cos'],[1.25]); cm=ls(aff(s['panel_c']['u']),s['panel_c']['v1']);
coords={'raw':x,'latent':z,'b_latent':aff(z)@bm,'c1_latent':c1(z),'c2_latent':c2(z),'c1_raw':c1(x),'c2_raw':c2(x),'caff_latent':aff(z)@cm,'caff_raw':aff(x)@cm}
folds=[(ix[:52],ix[52:])]+[(np.setdiff1d(ix,a),a) for a in np.array_split(ix,4)]
def gram_score(K,alpha):
 e=[]
 for a,b in folds:
  p=K[np.ix_(b,a)]@np.linalg.solve(K[np.ix_(a,a)]+alpha*np.eye(len(a)),Y[a]);d=abs(p-Y[b]);e.append([d[:,0].mean(),d[:,1:].mean()]);
 return np.array(e)
def poly(u,d):return np.column_stack([np.ones(len(u))]+[np.prod(u[:,j],axis=1) for k in range(1,d+1) for j in cw(range(u.shape[1]),k)])
results=[]
def add(name,K):
 for a in [1e-6,1e-4,.001,.01,.05]:
  e=gram_score(K,a);results.append((name,a,e[0],e[1:].mean(0),e[1:].max(0)))
for name,q in coords.items():
 for nn,w in [('id',q),('squash',q/np.sqrt(1+.09*q*q)),('tanh',np.tanh(q)),('sin',np.sin(q))]:
  for d in [1,2,3,4]:
   f=poly(w,d); add((name,nn,d),f@f.T)
 D=cdist(q,q,'sqeuclidean');
 for h in [.5,1,1.5,2,3,5]:add((name,'rbf',h),1+np.exp(-D/(2*h*h)))
for j in [0,1]:
 for name,a,e,avg,w in sorted(results,key=lambda r:r[-1][j])[:22]:print(j,name,a,e,avg,w)
