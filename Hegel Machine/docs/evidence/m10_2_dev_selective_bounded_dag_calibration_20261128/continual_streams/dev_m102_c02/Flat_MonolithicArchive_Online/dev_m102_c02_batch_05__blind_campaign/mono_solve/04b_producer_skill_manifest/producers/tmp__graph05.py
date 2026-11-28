import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from discover05 import *
from scipy.linalg import qr
from scipy.optimize import least_squares
from itertools import combinations
z=(np.r_[x,qa]-b)/g
np.set_printoptions(precision=6,suppress=True)
def projfit(u,v,fn,starts,bounds):
 def fit(p):
  h=A(fn(u,p)); coef=F(h,v); return h,coef
 def res(p):
  h,coef=fit(p);return (h@coef-v).ravel()
 opt=min((least_squares(res,st,bounds=bounds,gtol=2e-12,max_nfev=300) for st in starts),key=lambda o:o.cost)
 return opt.x,np.std(opt.fun),A(fn(z,opt.x))@fit(opt.x)[1]
fs={
 'powered':(lambda u,p:np.sign(u)*abs(u)**p[0]/(1+p[1]*abs(u)**p[2]),[[3,.22,4],[3,1,2],[5,50,3]],([.15,0,.1],[20,1000,20])),
 'cubic_power':(lambda u,p:u**3/(1+p[0]*abs(u)**p[1])**p[2],[[.22,4,1],[.22,2,2],[1,3,2]],([0,.1,.05],[100,16,20])),
 'mixrat':(lambda u,p:(u**3+p[0]*u)/(1+p[1]*abs(u)**p[2])**p[3],[[0,.22,4,1],[0,.22,2,2],[.05,1,3,2]],([-.5,0,.1,.05],[.5,100,16,20])),
 'positive_fixed':(lambda u,p:u**3/(1+.22*abs(u)**p[0])**p[1],[[2,1],[4,1],[5,2]],([.1,.05],[16,20])),
 'clipped':(lambda u,p:np.clip(u,-p[0],p[0])**3/(1+p[1]*abs(u)**p[2]),[[1.1,.22,2],[1.2,.22,4]],([.7,0,.1],[2,15,16])),
 'even':(lambda u,p:np.c_[u,abs(u)**p[0]/(1+p[1]*abs(u)**p[2])],[[2,.22,2],[2,.22,4]],([.5,0,.1],[7,15,16])),
}
w={}
for pn,vns in [('panel_b',['v']),('panel_c',['v1','v2'])]:
 for vn in vns:
  U,V=(s[pn][k] for k in ['u',vn]); name=pn[-1]+vn[-1]
  for ident,(fn,starts,bounds) in fs.items():
   if (vn=='v2')!=(ident=='even'):continue
   p,e,w[name+ident]=projfit(U,V,fn,starts,bounds); print(name,ident,p,e)
  for kind in ['powered','cubic']:
   w[name+'old'+kind]=m1._panel_operator({'u':U,'v':V},kind)(z)
w['z']=z
rng=np.random.default_rng(320); folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(25)]]
def score(h,rad=0):
 ans=[]
 for i,j in folds:
  if rad:
   scale=h[i].std(0); scale[scale<1e-12]=1; k=h[i]/scale; co=np.linalg.solve(k.T@k+rad*np.diag(np.r_[0,np.ones(k.shape[1]-1)]),k.T@y[i]); e=h[j]/scale@co-y[j]
  else:e=h[j]@F(h[i],y[i])-y[j]
  ans.append(np.mean(abs(e)))
 return np.r_[ans[0],np.mean(ans[1:]),np.max(ans[1:])]
def prod(d,ii):return np.column_stack([np.prod(d[:,inds],1) for inds in ii])
results=[]
def evaluate(name,d):
 ii=list(combinations_with_replacement(range(d.shape[1]),2)); jj=list(combinations_with_replacement(range(d.shape[1]),3))
 a=A(d); phi=prod(d,ii); psi=prod(d,jj)
 for n,h in [('linear',a),('paired',np.c_[a,d[:,0]*d[:,2],d[:,1]*d[:,3]]),('sum paired',np.c_[a,d[:,0]*d[:,2]+d[:,1]*d[:,3]]),('diff paired',np.c_[a,d[:,0]*d[:,2]-d[:,1]*d[:,3]]),('linear square',np.c_[a,d*d]),('quad',np.c_[a,phi]),('cubic',np.c_[a,phi,psi])]:
  results.append((score(h),name,n,0))
  if n=='cubic':
   for r in [.01,.1,1]: results.append((score(h,r),name,n,r))
for n,d in w.items():evaluate(n,d)
for bn in ['bvcubic_power','bvmixrat','bvoldpowered']:
 d=w[bn]; a=A(d); c1,c2=w['c1powered'],w['c2even']; q=np.c_[a,d[:,0]*d[:,2],d[:,1]*d[:,3]]
 for n,h in [('plus c1',np.c_[q,c1]),('plus c2',np.c_[q,c2]),('plus c1 c2',np.c_[q,c1,c2]),('plus z',np.c_[q,z]),('plus c1*c2',np.c_[q,c1*c2]),('plus c1+c2',np.c_[q,c1+c2]),('plus c1-c2',np.c_[q,c1-c2]),('plus c1**2',np.c_[q,c1**2]),('plus d*c1',np.c_[q,d*c1]),('plus d*c2',np.c_[q,d*c2])]:results.append((score(h),bn,n,0))
print('public, random avg, random worst');
for a in sorted(results,key=lambda p:p[0][1])[:32]:print(a)
for n in ['bvcubic_power','bvmixrat','bvoldpowered','c1powered','z']:
 d=w[n]; a=A(d); h=a[:52]; c=F(h,yt[:52]); resid=yt[:52]-h@c
 print('trajectory',n,np.mean(abs(a[52:72]@c-yt[52:72]),axis=0),np.std(resid),np.linalg.svd(yt[:52],compute_uv=False))
