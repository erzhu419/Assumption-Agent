import sys,pickle,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m100_c02_s00_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,g=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
o=m.solve_batch(d); tau,th=o['estimated_taus'],o['hereditary_weights']; past,hold=m._loads(cal,g)
combined={k:np.concatenate([te[k],g[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
xs=[m._features(tr,g)[-len(hold):],m._features(g,g)[:len(hold)],m._features(tr,combined)[-len(hold):],m._features(te,g)[-len(hold):]]
y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; F=np.c_[np.r_[-np.expm1(-t[:,None]/tau),[1/tau]],np.r_[t,1]]
for i,x in enumerate(xs):
 _,b,*_=m._design(np.log(tau),past,hold,x,dt,t); base=np.c_[b[:,:2]*th[:2]*tau,x@th[2:]]; e=y-base@F.T
 print('feature option',i,'max difference from train+diag',np.abs(x-xs[0]).max())
 for name,Z in [('last',np.c_[np.ones(len(x)),hold,past[:,-1]]),('history',np.c_[np.ones(len(x)),hold,past]),('history+x',np.c_[np.ones(len(x)),hold,past,x[:,1:]])]:
  ca=np.linalg.lstsq(Z,np.linalg.lstsq(F,e.T,rcond=None)[0].T,rcond=None)[0]; resid=e-Z@ca@F.T
  lev=np.sum(np.linalg.qr(Z,mode='reduced')[0]**2,axis=1); loo=resid/(1-lev[:,None]); print(name,'fit RMSE strain/rate',np.sqrt(np.mean(resid[:,:-1]**2)),np.std(resid[:,-1]),'LOO',np.sqrt(np.mean(loo[:,:-1]**2)),np.sqrt(np.mean(loo[:,-1]**2)))
  if name=='history+x': print('delta coeff rows 1 hold past0..5 feat0..3:',ca)
 if i==0:
  coeff=np.linalg.lstsq(F,y.T,rcond=None)[0].T
  print('observed rate minus shape fit',y[:,-1]-coeff@F[-1]); print('last past',past[:8,-1]);print('mean shape residual by time',np.mean(y-coeff@F.T,axis=0))
