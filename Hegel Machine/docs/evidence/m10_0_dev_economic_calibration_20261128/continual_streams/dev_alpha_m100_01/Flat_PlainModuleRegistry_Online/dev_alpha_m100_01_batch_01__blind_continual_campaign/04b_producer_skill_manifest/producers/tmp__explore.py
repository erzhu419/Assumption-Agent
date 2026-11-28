import pickle,sys,numpy as np
from scipy.linalg import lstsq
from base import _features,_bridge,_loads,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=160)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train'];te=d['test'];ch=p['campaign_diagnostics']; t=d['eval_times'];dt=d['past_dt']; br=_bridge(d['calibration_load_cells']); x=_features(tr,te);xc=_features(tr,ch)[len(tr['lab_id']):]; n=len(tr['lab_id']); pt,ht=_loads(tr,br);pc,hc=_loads(ch,br); qt=tr['strain_curves_obs']-tr['strain_0'][:,None];qc=ch['strain_curves_obs']-ch['strain_0'][:,None];rt=tr['instant_rate_obs'];rc=ch['instant_rate_obs']; taus,theta=_fit(pt,ht,x[:n],dt,t,qt,rt)
print('taus theta',taus,theta,'bridge',br)
def rms(x,axis=None):return np.sqrt(np.mean(x*x,axis=axis))
for a,b,c,q,r in [(pt,ht,x[:n],qt,rt),(pc,hc,xc,qc,rc)]:
 A,B=_design(taus,a,b,c,dt,t);ec=q-A@theta;er=r-B@theta
 print('curve rmse by time',rms(ec,0),'rate rmse',rms(er),'curve mae median',np.median(abs(ec)))
 if a is pc:
  basis=np.c_[-np.expm1(-t[:,None]/taus),t]; bb=np.vstack([basis,np.r_[1/taus,1]]); delta=lstsq(bb,np.c_[ec,er].T)[0].T; rem=np.c_[ec,er]-delta@bb.T
  print('free coefficients remaining rms',rms(rem,0)); print('baseline residual correlations vs past hold drift',np.corrcoef(np.c_[delta,pc,hc,xc].T)[:3,3:]);print('coeffs and initial residual',np.c_[delta,er]);print('curve median and percentiles',np.percentile(ec,[10,50,90],axis=0));print('diag hold',hc)
