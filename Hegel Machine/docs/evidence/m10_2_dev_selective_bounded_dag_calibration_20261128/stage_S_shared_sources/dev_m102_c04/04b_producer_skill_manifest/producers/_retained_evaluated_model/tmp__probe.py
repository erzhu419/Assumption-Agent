import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; print('times',s['eval_times']); n=len(x); split=52

def features(x,d):
 return np.column_stack([np.prod(x[:,idx],axis=1) for i in range(d+1) for idx in combinations_with_replacement(range(x.shape[1]),i)])
def affine(a,v):return np.linalg.lstsq(features(a,1),v,rcond=None)[0]
a=s['panel_a']; ea=np.linalg.norm(features(a['u'],1)@affine(a['u'],a['v'])-a['v'],axis=1); good=ea<2; da=affine(a['u'][good],a['v'][good]); xa=(x-da[-1])/np.diag(da[:4]); xf=features(x,1)@da
b=s['panel_b']; db=affine(b['u'],b['v']); print('B singular',np.linalg.svd(db[:4],compute_uv=False));
c=s['panel_c']; u=c['u']; co={}
for key in ('v1','v2'):
 v=c[key]; print(key,'range',v.min(0),v.max(0));
 for name,fun,inv in [('identity',lambda a:a,lambda a:a),('tanh',np.tanh,lambda a:np.arctanh(np.clip(a,-.99999,.99999))),('sin',np.sin,lambda a:np.arcsin(np.clip(a,-.99999,.99999))),('softsign',lambda a:a/(1+abs(a)),lambda a:a/(1-np.minimum(abs(a),.99999)))]:
  for d in (1,2):
   f=features(u,d); w=np.linalg.lstsq(f,inv(v),rcond=None)[0]; w=least_squares(lambda q:(fun(f@q.reshape(w.shape))-v).ravel(),w.ravel(),max_nfev=100).x.reshape(w.shape); print(key,name,d,round(np.mean(abs(fun(f@w)-v)),6));
   if name=='identity' and d==2: print('quad weights',np.round(w,3));co[key]=(fun,w)
_,_,vv=np.linalg.svd(z[:52],full_matrices=False); zz=z@vv[:2].T; print('mode vectors',np.round(vv[:2],4))
for name,q in [('x',x),('Ainv',xa),('Afwd',xf),('B(x)',features(x,1)@db),('B(Ainv)',features(xa,1)@db),('B(Afwd)',features(xf,1)@db),('Binv(Ainv)',(xa-db[-1])@np.linalg.inv(db[:4]))]:
 scores=[]
 for g in [lambda q:q,np.tanh,np.sin,lambda q:q/(1+q*q),lambda q:q/np.sqrt(1+q*q),lambda q:np.exp(-q*q)]:
  for d in (1,2,3):
   f=np.c_[features(q,1),features(g(q),d)[:,1:]]; w=np.linalg.lstsq(f[:split],np.c_[y,zz][:split],rcond=1e-9)[0]; e=f[split:]@w-np.c_[y,zz][split:]; scores.append((g.__name__,d,*np.round([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@vv[:2]))],5)))
 print(name,sorted(scores,key=lambda a:a[2])[:7]); print('state',sorted(scores,key=lambda a:a[3])[:3]);
