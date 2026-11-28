import pickle, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; T=s['eval_times']; Y=d['traj']; tr=np.arange(52); vv=np.arange(52,72)
bases={'exp+lin':lambda p:np.c_[-np.expm1(-p[0]*T),T], 'twoexp':lambda p:np.c_[-np.expm1(-p[0]*T),-np.expm1(-p[1]*T)], 'exp+constant':lambda p:np.c_[-np.expm1(-p[0]*T),np.ones(len(T))], 'repeated':lambda p:np.c_[-np.expm1(-p[0]*T),T*np.exp(-p[0]*T)], 'exp+sin':lambda p:np.c_[-np.expm1(-p[0]*T),np.sin(p[1]*T)], 'euler+lin':lambda p:np.c_[-np.expm1(-p[0]*(T-T[0]+p[1])),T]}
for name,f in bases.items():
 k=2 if name in ('twoexp','exp+sin','euler+lin') else 1
 def res(p):
  a=f(p); return (Y[tr]-(a@np.linalg.lstsq(a,Y[tr].T,rcond=None)[0]).T).ravel()
 p=min((least_squares(res,p,bounds=(.001,8.)) for p in ([.7]*k,[1.2,.2][:k],[2.,1.25][:k],[.22,2.15][:k])),key=lambda r:r.cost).x; a=f(p); B=np.linalg.lstsq(a,Y.T,rcond=None)[0].T; print(name,p,np.mean(abs(Y[vv]-B[vv]@a.T)),np.mean(abs(res(p))),np.round(np.corrcoef(np.c_[d['x'],d['y'],B].T)[-2:],3)); print('coeff ranges',B.min(axis=0),B.max(axis=0))
U,D,V=np.linalg.svd(Y[tr],full_matrices=False); print('time modes',V[:3]); print('svd errors',[(k,np.mean(abs(Y[vv]-(Y[vv]@V[:k].T)@V[:k]))) for k in (1,2,3)]); print('summary clean/noisy',d['summary'],[d['y'].mean(),d['y'].std(),Y[:,-1].mean(),Y[:,-1].std()])
