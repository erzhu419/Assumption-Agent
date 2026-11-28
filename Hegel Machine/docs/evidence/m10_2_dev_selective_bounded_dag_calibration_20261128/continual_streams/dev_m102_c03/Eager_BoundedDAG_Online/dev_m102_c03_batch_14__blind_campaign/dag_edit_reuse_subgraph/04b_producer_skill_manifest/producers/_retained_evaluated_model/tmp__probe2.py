import itertools as it,numpy as np,pickle,runpy
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/node_01.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['panel_c']; u,v=c['u'],c['v2']; ls=m['_ls']; bias=m['_bias']; prod=u*np.roll(u,1,axis=1); base=bias(np.c_[u,prod]); R=v-base@ls(base,v); print('C2 svd',np.linalg.svd(R,compute_uv=False)); print('base coefs',np.round(ls(base,v),3));
funcs={'id':lambda d,p:d,'cos':lambda d,p:np.cos(p*d),'sin':lambda d,p:np.sin(p*d),'tanh':lambda d,p:np.tanh(p*d),'rat':lambda d,p:d/(1+p*d*d),'square':lambda d,p:d*d,'gauss':lambda d,p:np.exp(-p*d*d)}
res=[]
for lag in (1,-1,2):
 w=np.roll(u,lag,axis=1)
 for pair,d in [('self',u),('sum',u+w),('diff',u-w),('prod',u*w),('sqsum',u*u+w*w),('dir',u*w*w),('triple',u*w*np.roll(u,2,axis=1))]:
  for name,fn in funcs.items():
   for kind in ('add','shared','denom','sep'):
    for ini in (1,2.15,3.5):
     def design(p):
      if kind=='add':return np.c_[base,fn(d,p[0])]
      if kind=='sep':return bias(np.c_[u,fn(d,p[0])])
      if kind=='shared':return bias(u+p[1]*prod+p[2]*fn(d,p[0]))
      return bias((u+p[1]*prod)/(1+p[0]*d*d))
     def fun(p):
      f=design(p);return (f@ls(f,v)-v).ravel()
     rr=least_squares(fun,[ini,.3,.25],bounds=([.01,-3,-3],[8,3,3]),max_nfev=45); e=rr.fun.reshape(v.shape); f=design(rr.x); lev=np.sum(f*np.linalg.pinv(f).T,axis=1); res.append((np.std(e),np.sqrt(np.mean((e/(1-lev[:,None]))**2)),lag,pair,name,kind,rr.x))
for rr in sorted(res,key=lambda x:x[1])[:20]:print(rr)
