import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import alpha_w04_batch_01_solver as s
with open('/holdout_task/public_alpha_w04_batch_01.pkl','rb') as f: a=pickle.load(f)[0]
d,g=a['dataset'],a['gt']; tr,te=d['train'],d['test']; T=d['eval_times']; dt=d['past_dt']; aff=s._calibrate(d['calibration_load_cells'])
def sub(u,m): return {k:v[m] for k,v in u.items()}
def miss(u): return ~np.isfinite(u['coupon_features_raw']).all(axis=1)
def stat(r): return np.sqrt(np.mean(r*r))
def mad(r): return max(1e-6,1.4826*np.median(abs(r-np.median(r))))
def calc(train,test):
 x,xt=s._align(train,test); h,p=s._stress(train,aff); ht,pt=s._stress(test,aff)
 y=train['strain_curves_obs']-train['strain_0'][:,None]; rate=train['instant_rate_obs']; init=train['initial_memory_states']; m=~miss(train)
 tau,w=s._fit(dt,T,h[m],p[m],x[m],init[m],y[m],rate[m]); labs=train['lab_id']; lt=test['lab_id']; it=test['initial_memory_states']; mt=miss(test)
 # Estimate each coupon's drift from its intact curve samples after subtracting relaxation.
 b,r=s._design(tau,dt,T,h,p,x,init)
 drift=np.median((y[:,T>=1.]-b[:,T>=1.,:2]@w[:2])/T[T>=1.],axis=1)
 source=r[:,2:]@w[2:]; delta=drift-source
 bias=np.zeros(4); slope=np.zeros(4)
 for lab in np.unique(labs):
  k=(labs==lab)&m
  xx=source[k]-np.median(source[k]); yy=delta[k]; scale=max(mad(yy),0.003)
  off,sl=least_squares(lambda z:(yy-z[0]-xx*z[1])/scale, [0.,0.],loss='soft_l1').x
  bias[lab]=off-sl*np.median(source[k]); slope[lab]=sl
 print('tau',np.round(tau,5),'lab drift bias',np.round(bias,5),'slope adjustment',np.round(slope,5))
 bt,rt=s._design(tau,dt,T,ht,pt,xt,it); base=test['strain_0'][:,None]+bt@w; rate_t=test['instant_rate_obs']; drift_t=rt[:,2:]@w[2:]
 return tau,pt,[(tag,base+T*(adjust+np.where(mt,1.,mix)*(rate_t-rt@w-adjust))[:,None]) for tag,adjust in [('shared',np.zeros(len(mt))),('lab',bias[lt]+slope[lt]*drift_t)] for mix in (0.,.25,.5)]
def report(train,test,y,truth=False):
 tau,hold,out=calc(train,test); mask=miss(test); lab=test['lab_id'];
 for tag,pred in out:
  j=[np.ones(len(y),bool),mask,~mask]+[lab==i for i in np.unique(lab)]; e=pred-y
  if truth:
   den=np.maximum(abs(hold),.5); sec=(pred[:,-1]-test['strain_0'])/den-g['te_secant']; haz=s._crossing(T,test['strain_0'],pred,test['critical_strain'])-g['te_cross']; print(tag,'strain/secant/hazard overall, missing, worst lab', np.round([[stat(v),stat(v[mask]),max(stat(v[k]) for k in j[3:])] for v in (e,sec,haz)],5).tolist())
  else:
   print(tag,'curve median/terminal MAE complete and missing',np.round([(np.median(abs(e[k])),np.mean(abs(e[k,-1]))) for k in j[:3] if k.any()],5).tolist())
for lab in range(4):
 m=tr['lab_id']==lab; print('HELD LAB',lab); report(sub(tr,~m),sub(tr,m),tr['strain_curves_obs'][m])
fold=np.zeros(len(tr['lab_id']),int); rng=np.random.default_rng(17)
for lab in range(4):
 for z in (False,True):
  ix=np.flatnonzero((tr['lab_id']==lab)&(miss(tr)==z)); fold[rng.permutation(ix)]=np.arange(len(ix))%4
for k in range(4):
 m=fold==k; print('HELD COUPONS',k); report(sub(tr,~m),sub(tr,m),tr['strain_curves_obs'][m])
print('PUBLIC'); report(tr,te,g['te_clean'],True)
