import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; x,y,z=T['x'],T['y'],T['traj']; t=s['eval_times']
def lin(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return np.column_stack([np.prod(x[:,c],axis=1) for k in range(d+1) for c in combinations_with_replacement(range(x.shape[1]),k)])
def fit(A,Y):return np.linalg.lstsq(A,Y,rcond=None)[0]
def rms(e):return np.mean(e**2)**.5
a=s['panel_a']; A=lin(a['u']); B=fit(A,a['v']); B=least_squares(lambda b:(A@b.reshape(5,4)-a['v']).ravel(),B.ravel(),loss='cauchy',f_scale=.02).x.reshape(5,4); err=np.linalg.norm(A@B-a['v'],axis=1); print('A robust',B,'sorted row err',np.round(np.sort(err),4)); print('ranges x/tr_test/panels',[(np.min(v,axis=0),np.max(v,axis=0)) for v in [x,s['test']['x']]+[s[k]['u'] for k in ('panel_a','panel_b','panel_c')]])
bas=lambda r:-np.expm1(-t[:,None]*r)
r=least_squares(lambda r:(bas(r)@fit(bas(r),z[:52].T)-z[:52].T).ravel(),[1.6,.3],bounds=([.58,.01],[5,1.2])).x; Q=np.linalg.qr(bas(r))[0]; print('traj rates, residual',r,rms(z-z@Q@Q.T)); print('traj coefficients sample',fit(bas(r),z.T)[:,:8]); Y=np.c_[y,z@Q]
for k in ['panel_b','panel_c']:
 p=s[k]; u=p['u'];
 for name in p.keys()-{'u'}:
  v=p[name]; target={ 'raw':v,'atanh':np.arctanh(np.clip(v,-.999999,.999999)), 'asinh':np.arcsinh(v), 'asin':np.arcsin(np.clip(v,-1,1))}; print(k,name,'min/max',v.min(axis=0),v.max(axis=0));
  for lab,w in target.items():
   out=[]
   for f in (lin(u),poly(u,2),lin(np.sin(u)),np.c_[lin(u),u**2],np.c_[lin(u),u**3],lin(np.tanh(u)),np.c_[lin(u),np.sin(u)],np.c_[lin(u),np.cos(u)]):
    h=np.diag(f@np.linalg.pinv(f)); e=(w-f@fit(f,w))/(1-h[:,None]); out.append(rms(e))
   print(lab,np.round(out,5))
for name,U in [('raw',x),('Aforward',lin(x)@B),('Ainverse',(x-B[0])@np.linalg.inv(B[1:]))]:
 for fn in [lambda u:poly(u,2),lambda u:poly(u,3),lambda u:np.c_[poly(u,2),np.sin(u),np.cos(u)],lambda u:np.c_[poly(u,2),np.tanh(u)],lambda u:np.c_[lin(u),u**2,np.sin(u),np.cos(u)]]:
  F=fn(U); E=F[52:]@fit(F[:52],Y[:52])-Y[52:]; print('reg',name,F.shape[1],np.abs(E[:,0]).mean(),np.abs(E[:,1:]@Q.T).mean())
