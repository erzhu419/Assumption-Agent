import sys,pickle,importlib.util,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
s=importlib.util.spec_from_file_location('prev','/tmp/hegel_continual_archive/alpha_m100_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=calibrate_upstream(dict(i['dataset'], chamber_verification_trials=i['campaign_diagnostics'])); tr,te,q=[d[k] for k in ('train','test','chamber_verification_trials')];dt,t=d['past_dt'],d['eval_times'];
def g(s): return s['past_channel_raw'].sum(-1),s['hold_channel_raw'].sum(-1)
def f(x):return np.column_stack((np.ones(len(x)),x))
xtr,xte=m._features(tr,te); taus,th=m._fit(*g(tr),f(xtr),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);print('taus theta',taus,th)
def y(s):return np.column_stack((s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']))
curve,rate=m._design(taus,*g(tr),f(xtr),dt,t); print('train abs resid q',np.quantile(abs(y(tr)-np.column_stack((curve@th,rate@th))),[.5,.9,.95,.99,1],axis=0).round(4))
B=np.row_stack((np.column_stack((-np.expm1(-t[:,None]/taus),t)),np.r_[1/taus,1]));Y=y(q); A=Y@np.linalg.pinv(B.T); print('diag row mode residual',np.round(np.sqrt(np.mean((Y-A@B.T)**2,axis=1)),4));print('A range',np.quantile(A,[0,.25,.5,.75,1],axis=0));
merge=lambda a,b:{k:np.concatenate((a[k],b[k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
xqs=[m._features(merge(tr,te),q)[1], m._features(tr,q)[1], m._features(q,q)[1]]
p,h=g(q); age=np.cumsum(dt[::-1])[::-1]-dt;z=p@(np.exp(-age[:,None]/taus)*-np.expm1(-dt[:,None]/taus));
for j,xq in enumerate(xqs):
 dr=f(xq)@th[2:]; base=np.column_stack(((h[:,None]-z)*th[:2],dr)); reset=np.column_stack((h[:,None]*th[:2],dr)); print('ref mode',j,'drift std',dr.std());
 for name,C in [('base',base),('no_drift',base*[1,1,0]),('reset',reset),('reset+drift_mix',np.column_stack((reset[:,:2]+dr[:,None]*[.19552,1.90306],.01715*dr)))]:
  e=Y-C@B.T; print(name, 'median/90 curve',np.quantile(abs(e[:,:-1]),[.5,.9]), 'rate',np.quantile(abs(e[:,-1]),[.5,.9]));
 if j==0:
  names=['1','h','z0','z1','dr','x0','x1','x2','x3','p0','p1','p2','p3','p4','p5']; M=np.column_stack((np.ones(len(h)),h,z,dr,xq,p)); print('corr\n',[(n,np.round(a,3)) for n,a in zip(names[1:],np.corrcoef(np.column_stack((M[:,1:],A)).T)[:-3,-3:])]);print('amps rows:\n',np.round(np.column_stack((h,z,dr,A)),4));
