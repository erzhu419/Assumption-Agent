exec(open('/tmp/chamber_law.py').read().split('for label,F in')[0])
import importlib.util
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m2=importlib.util.module_from_spec(s); s.loader.exec_module(m2)
g=theta[2]+xc@theta[3:]; F=np.c_[np.ones(n),hc,g]; lab=c['lab_id']; miss=np.isnan(c['coupon_features_raw']).any(1); ids=np.arange(n); w=.5
for mode in ['free','derivative','rate_fixed','hold_fixed','hold_and_rate_fixed']:
 def fit(ix):
  def pred(p):
   tt=tau*np.exp(p[:2]); A=p[2:8].reshape(2,3).copy(); D=np.r_[p[8],0,p[9]]
   if mode in ['hold_fixed','hold_and_rate_fixed']: A[:,1]=theta[:2]
   if mode in ['derivative','rate_fixed','hold_and_rate_fixed']:
    dr=np.r_[0.,sum(theta[:2]/tau),1.]; D=dr-A.T@(1/tt)
   amp=F@A.T; drift=F@D; C=amp@(-np.expm1(-t[:,None]/tt)).T+drift[:,None]*t; R=amp@(1/tt)+drift
   if mode=='rate_fixed': R=hc*sum(theta[:2]/tau)+g
   return np.c_[C,w*R]
  law=m2._CHAMBER_LAW; p0=np.r_[np.log(law[8:]),law[:8]]; yy=np.c_[y,w*v]; q=least_squares(lambda p:(pred(p)[ix]-yy[ix]).ravel(),p0,loss='cauchy',f_scale=.004,max_nfev=100).x
  return pred(q)-yy,q
 for scope in ['lab','coupon']:
  e=np.zeros((n,len(t)+1)); labels=lab if scope=='lab' else ids
  for k in np.unique(labels):
   mask=labels==k; e[mask]=fit(ids[~mask])[0][mask]
  print(mode,scope,'all',report(e),'complete',report(e[~miss]))
 print('params',fit(ids)[1].round(6))
base=m2.solve_batch(d); print('deployed shape/rate finite',base['predicted_strain'].shape,base['initial_rate_pred'].shape,np.isfinite(base['predicted_strain']).all(),np.isfinite(base['initial_rate_pred']).all()); print('calibrated hold ranges diagnostic/test',np.quantile(hc,[0,1]),np.quantile(m2._loads(te,b)[1],[0,1])); print('in-sample fit',report(np.column_stack((m2._chamber(tau,theta,xc,hc,t,m2._CHAMBER_LAW)[0]-y,w*(m2._chamber(tau,theta,xc,hc,t,m2._CHAMBER_LAW)[1]-v)))))
