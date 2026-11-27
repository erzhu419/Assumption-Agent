import pickle, sys
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m99_c04_s00_solver as m
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=item['dataset']; q=item['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; s=m._stress_map(d['calibration_load_cells'])
x,_=m._features(tr,te); tau,th=m._fit(tr,x,s,dt,t)
_,xq=m._features(tr,q); xlocal,_=m._features(q,q)
p=s[0]+q['past_channel_raw']@s[1:]; h=s[0]+q['hold_channel_raw']@s[1:]; age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; z=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-age[:,None]/tau)); gap=h[:,None]-z
F={'hold':h,'z0':z[:,0],'z1':z[:,1],'g0':gap[:,0],'g1':gap[:,1],'g0*g1':gap.prod(1),'g0-g1':gap[:,0]-gap[:,1],'lab':q['lab_id'],'stratum':q['stratum']}
for i in range(p.shape[1]): F[f'p{i}']=p[:,i]; F[f'h-p{i}']=h-p[:,i]
for i in range(4): F[f'x{i}']=xq[:,i]; F[f'rawx{i}']=np.nan_to_num(q['coupon_features_raw'][:,i])
from itertools import combinations, combinations_with_replacement
_, xall = m._features(tr, {k:np.concatenate([te[k],q[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')})
U=np.column_stack([-np.expm1(-t),t]); Ua=np.vstack([U,[1,1]])
for name,xx in [('own',xlocal),('train+diag',xq),('all',xall[-len(h):])]:
 c,r=m._design(q,xx,s,dt,t,tau); E=q['strain_curves_obs']-q['strain_0'][:,None]-c@th; er=q['instant_rate_obs']-r@th
 B=np.linalg.lstsq(Ua,np.column_stack([E,er]).T,rcond=None)[0].T
 print(name,'A and slope',np.c_[B,B.sum(1)].round(4).tolist() if name=='own' else '', 'initial correction rms',np.std(B.sum(1)), 'rate rms',np.sqrt(np.mean(er**2)))
 def score(X,y):
  b=np.linalg.lstsq(X,y,rcond=None)[0]; res=y-X@b; loo=res/np.maximum(1e-8,1-(X*np.linalg.pinv(X).T).sum(1)); return np.sqrt(np.mean(res**2)),np.sqrt(np.mean(loo**2)),b
 def dictionary(x):
  V={f'x{i}':x[:,i] for i in range(4)}; V.update({'h':h,'z0':z[:,0],'z1':z[:,1],'g0':gap[:,0],'g1':gap[:,1]}); V.update({f'p{i}':p[:,i] for i in range(p.shape[1])}); return V
 V=dictionary(xx); base=list(V)
 for i,j in combinations_with_replacement(base,2): V[i+'*'+j]=V[i]*V[j]
 for i,j,k in combinations_with_replacement([f'x{i}' for i in range(4)],3): V[i+'*'+j+'*'+k]=V[i]*V[j]*V[k]
 for i,j in combinations([f'x{i}' for i in range(4)],2):
  for sg in (1,-1):
   v=V[i]+sg*V[j]; key=f'({i}{sg:+}{j})'; V['tanh'+key]=np.tanh(v); V[key+'/(1+v2)']=v/(1+v*v)
 X=np.column_stack([np.ones(len(xx)),xx]); print('linear A, drift, sum',[(x,y,b.round(5).tolist()) for x,y,b in [score(X,B[:,0]),score(X,B[:,1]),score(X,B.sum(1))]])
 for y in (B[:,0],B.sum(1)):
  best=sorted([(score(np.column_stack([X,v]),y)[1],k) for k,v in V.items()]); print('best added',best[:5]); M=np.ones((len(xx),1)); chosen=[]
  for _ in range(5):
   val,k=min((score(np.column_stack([M,v]),y)[1],k) for k,v in V.items() if k not in chosen); chosen.append(k); M=np.column_stack([M,V[k]]); print('sparse',chosen[-1],round(val,6),score(M,y)[2].round(5).tolist())
