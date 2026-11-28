import pickle, sys
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from batch_solver import _coordinates, _fit, _scalar_design, _trajectory
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; tr=s['train']; raw=tr['x']; x=_coordinates(s,raw); y=tr['y']; z=tr['traj']; n=len(x)
aff=lambda a: np.column_stack((np.ones(len(a)),a))
quad=lambda a: np.column_stack([aff(a)]+[a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(4),2)])
q1=x[:,0]*x[:,2]; q2=x[:,1]*x[:,3]; sd=_scalar_design(x)
ys={'tied':sd,'tied+offset':aff(sd),'untied':np.c_[x,q1,q2],'untied+offset':aff(np.c_[x,q1,q2]),'raw quadratic':quad(raw)}
for k,d in ys.items():
 print(k,'fit',np.round(_fit(d,y),6),'RMSE',np.mean((d@_fit(d,y)-y)**2)**.5)
for name,holdouts in [('blocked',[np.arange(i,min(i+16,n)) for i in range(0,n,16)]),('random',np.array_split(np.random.default_rng(7).permutation(n),5)),('extreme radius',[np.argsort(np.linalg.norm(x,axis=1))[-16:]]),('axis tails',[np.argsort(x[:,j])[sl] for j in range(4) for sl in (slice(0,12),slice(-12,None))])]:
 errors={k:[] for k in ys}; states={k:[] for k in ['raw affine','operator linear','two-mode','two-mode affine']}
 for va in holdouts:
  it=np.setdiff1d(np.arange(n),va)
  for k,d in ys.items(): errors[k].append(np.mean(abs(y[va]-d[va]@_fit(d[it],y[it]))))
  for k,p in [('raw affine',aff(raw[va])@_fit(aff(raw[it]),z[it])),('operator linear',x[va]@_fit(x[it],z[it])),('two-mode',_trajectory(x[it],z[it],x[va],t)),('two-mode affine',_trajectory(aff(x[it]),z[it],aff(x[va]),t))]: states[k].append(np.mean(abs(z[va]-p),axis=0))
 print(name,'scalar mean/max',[(k,round(np.mean(v),6),round(np.max(v),6)) for k,v in errors.items()]); print(name,'state mean/max-timestamp',[(k,round(np.mean(v),6),round(np.max(v),6)) for k,v in states.items()]); print('trajectory terminal RMSE proxy MAE by model',[(k,round(np.mean(np.array(v)[:,-1]),6)) for k,v in states.items()])
va=np.arange(56,n);it=np.arange(56);print('last16 candidate',np.mean(abs(y[va]-sd[va]@_fit(sd[it],y[it]))),np.mean(abs(z[va]-_trajectory(x[it],z[it],x[va],t)),axis=0))
b=lambda p:-np.expm1(-t[:,None]*p)
o=least_squares(lambda p:(b(p)@_fit(b(p),z.T)-z.T).ravel(),[.4,1.8]); print('rates',o.x,'operator loadings',_fit(x,_fit(b(o.x),z.T).T)); print('trajectory residual std',np.std(z-_trajectory(x,z,x,t),axis=0))
