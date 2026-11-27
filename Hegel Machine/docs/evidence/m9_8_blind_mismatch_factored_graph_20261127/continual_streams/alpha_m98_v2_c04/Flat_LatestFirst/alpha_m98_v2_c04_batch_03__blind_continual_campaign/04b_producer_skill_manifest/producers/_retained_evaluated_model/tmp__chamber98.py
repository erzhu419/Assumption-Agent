import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c04_s00_solver as s
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; dg=item['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; br=d['calibration_load_cells']; raw=br['bridge_raw']; true=br['true_channel_loads']; b=s._bridge(br)
xb=s._features(tr,te); x=xb[:len(tr['lab_id'])]; p,h=s._loads(tr,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,theta=s._fit(x,p,h,dt,t,y,r)
print('tau',tau,'theta',theta); print('linear bridge rms, max by channel',np.sqrt(np.mean((np.c_[np.ones(len(raw)),raw]@b-true)**2,0)),np.max(np.abs(np.c_[np.ones(len(raw)),raw]@b-true),0))
for k in br['single_channel_sweeps']: print('sweep',k,np.shape(br['single_channel_sweeps'][k]),np.asarray(br['single_channel_sweeps'][k]).ravel()[:10])
for name in ['ref_coupons','lab_id','stratum','coupon_features_raw']:
 pass
joint={k:np.concatenate([te[k],dg[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}; xd=s._features(tr,joint)[-len(dg['lab_id']):]
p,h=s._loads(dg,b); yy=dg['strain_curves_obs']-dg['strain_0'][:,None]; rr=dg['instant_rate_obs']; a,f=s._design(xd,p,h,dt,t,tau); e=yy-(a@theta).reshape(yy.shape); er=rr-f@theta
u,sv,v=np.linalg.svd(e,full_matrices=False); print('residual singular values',sv.round(5)); print('leading shape normalized', (v[0]/v[0,-1]).round(5)); print('scaled powers', (t/t[-1])**2)
later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; mem=-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau); z=p@mem; gap=h[:,None]-z; drift=theta[2]+xd@theta[3:];
coef=np.linalg.lstsq(np.column_stack([t,t*t,t*t*t]),e.T,rcond=None)[0].T
print('lab stratum hold memories drift rate_resid poly1 poly2 poly3, x:'); print(np.round(np.c_[dg['lab_id'],dg['stratum'],h,z,drift,er,coef,xd],4))
base=np.c_[h,p,z,gap,xd,drift,rr,dg['strain_0']]; names=['hold']+['past'+str(i) for i in range(p.shape[1])]+['z0','z1','gap0','gap1','x0','x1','x2','x3','drift','rate','strain0']; out=np.c_[er,coef,e[:,-1]]
for i in range(out.shape[1]):
 cand=[]
 for j,n in enumerate(names):
  for fcn in [lambda x:x,np.abs,np.square]:
   x1=fcn(base[:,j]); cr=np.corrcoef(x1,out[:,i])[0,1]; cand.append((round(abs(cr),3),round(cr,3),n,str(fcn) if fcn in [np.abs,np.square] else 'linear'))
 print('strongest',i,sorted(cand,reverse=True)[:7])
for j in range(4):
 rows=dg['lab_id']==j
 print('diag feature panel',j, np.median(dg['ref_coupons'][rows].mean(0)),np.median(dg['ref_coupons'][rows].std(0)))
