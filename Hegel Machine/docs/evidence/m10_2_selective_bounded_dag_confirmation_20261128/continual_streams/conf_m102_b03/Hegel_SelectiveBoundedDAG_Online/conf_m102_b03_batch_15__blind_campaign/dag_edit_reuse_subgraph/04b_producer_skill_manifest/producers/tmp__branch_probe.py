import pickle,runpy,itertools,numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/node_00.py'); orig=m['run_node']({'__input__':s}); sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; ls=m['_ls']; one=m['_ones']; u=s['panel_c']['u']; z=orig['z']; n=len(s['train']['x']); cut=n-20
fs={'softsign':lambda a:a/(1+abs(a)), 'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'atan':np.arctan,'erf':lambda a:2*expit(1.6*a)-1,'sq':lambda a:a*a,'rational':lambda a:a/(1+a*a)}
def fit(u,v,fun,start,mix=True):
 def d(x,p): return one(np.c_[x,fun(x*p[0]+p[1:])] if mix else fun(x*p[0]+p[1:]))
 def resid(p):
  a=d(u,p); return (a@ls(a,v)-v).ravel()
 p=least_squares(resid,[start,0,0,0,0],bounds=([.03,-8,-8,-8,-8],[12,8,8,8,8]),max_nfev=300).x
 return np.mean(abs(resid(p))),p,ls(d(u,p),v),d
best=[]
for k in ('v1','v2'):
 v=s['panel_c'][k]; res=[]
 for name,fun in fs.items():
  t=min((fit(u,v,fun,start) for start in (.65,.9,1.25,1.9,2.4,3.4)),key=lambda t:t[0]); res.append((name,t))
 res.sort(key=lambda a:a[1][0]); print(k,[(name,round(t[0],7),np.round(t[1],4)) for name,t in res]); name,t=res[0]; h=[]
 for hold in np.array_split(np.random.default_rng(23).permutation(len(u)),4):
  a=np.setdiff1d(np.arange(len(u)),hold); _,p,w,d=fit(u[a],v[a],fs[name],t[1][0]); h.append(np.mean(abs(d(u[hold],p)@w-v[hold])))
 print('panel holdout',h); e,p,w,d=t; best.append((d(z,p)@w,fs[name](z*p[0]+p[1:])))
q,qb=best[0]; r,rb=best[1]; keep=slice(0,n); st=dict(s,train={k:(v[:cut] if k in ('x','y','traj') else v) for k,v in s['train'].items()},test={'x':s['train']['x'][cut:]})
for qn,qp in [('original',orig['product']),('branch',q),('basis',qb),('branch+basis',np.c_[q,qb]),('branch+z',np.c_[q,z]),('original+branch',np.c_[orig['product'],q])]:
 for rn,rp in [('original',orig['cosine']),('branch',r),('basis',rb),('branch+basis',np.c_[r,rb]),('branch+z',np.c_[r,z]),('original+branch',np.c_[orig['cosine'],r])]:
  f=dict(orig,Q=q,R=r,product=qp,cosine=rp); f={k:v[keep] for k,v in f.items()}; o=sink({'__input__':st,'node_00':f}); print(qn,rn,round(np.mean(abs(o['y_hat']-s['train']['y'][cut:])),6),round(np.mean(abs(o['state_pred']-s['train']['traj'][cut:])),6))
