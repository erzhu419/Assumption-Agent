import sys,pickle,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m100_c03_batch_03_solver as m
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=p['dataset']; d=p['campaign_diagnostics']; tr,te=ds['train'],ds['test']; dt,t=ds['past_dt'],ds['eval_times']; n=len(tr['lab_id']); nd=len(d['lab_id']); x=m._features(tr,te); xd=m._features(tr,d)[n:]; b=m._bridge(ds['calibration_load_cells']); past,h=m._loads(tr,b); tau,th=m._fit(x[:n],past,h,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); past,h=m._loads(d,b); c,r=m._design(tau,xd,past,h,dt,t); dr=th[2]+xd@th[3:]; y=np.column_stack((d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs'])); base=np.column_stack((c@th,r@th)); obs_time=np.r_[t,0]; print('panel labs',np.unique(d['lab_id'],return_counts=True),'fit',tau,th); print('worst baseline',np.median(abs(y-base),0).max(), 'rate',np.median(abs(y[:,-1]-base[:,-1]))); print('residual singular values',np.linalg.svd(y-base,compute_uv=False).round(4)); print('residual medians',np.median(y-base,0).round(4));
def et(q): return np.r_[-np.expm1(-t/q),1/q]
def show(label,pred):
 e=y-pred; print(label,'median/worst horizon/rate/full RMSE',np.median(abs(e[:,:-1])),np.median(abs(e[:,:-1]),0).max(),np.median(abs(e[:,-1])),np.sqrt(np.mean(e**2)))
for drift in ('linear','buildup','free'):
 D=np.r_[t,1] if drift!='buildup' else et(1); Y=y-dr[:,None]*D if drift!='free' else y
 for free_tau in (False,True):
  def temporal(q): return np.vstack([et(a) for a in q]+([D] if drift=='free' else []))
  def proj(q):
   A=temporal(q); coef=np.linalg.lstsq(A.T,Y.T,rcond=None)[0].T; return coef@A-Y,coef
  q=tau if not free_tau else np.sort(np.exp(least_squares(lambda z: proj(np.exp(z))[0].ravel(),np.log(tau),bounds=(np.log(.06),np.log(70))).x)); e,a=proj(q); show((drift,q.round(5)),y+e)
  if not free_tau or drift=='free': continue
  np.set_printoptions(precision=3,suppress=True); cols=np.column_stack((h,past,c[:,0,:2]/(-np.expm1(-t[0]/tau)),dr,xd)); corr=np.corrcoef(np.column_stack((a,cols)),rowvar=False)[:2,2:]; print('amp correlation hold/past/gaps/dr/features',corr); print('first 8 rows [hold,gaps,amp,dr]',np.column_stack((cols[:,[0,7,8]],a,dr))[:8]);
