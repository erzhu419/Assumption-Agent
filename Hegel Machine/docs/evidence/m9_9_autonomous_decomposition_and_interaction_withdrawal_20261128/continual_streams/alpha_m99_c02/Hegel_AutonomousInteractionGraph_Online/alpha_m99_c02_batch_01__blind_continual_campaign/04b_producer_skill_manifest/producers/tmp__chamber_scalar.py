import numpy as np, sys, pickle
sys.path.insert(0,'/tmp/hegel_continual_archive'); import alpha_m99_c02_s00_solver as s
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,di=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; b=s._bridge(d['calibration_load_cells']); x,xt=s._features(tr,te); _,xd=s._features(tr,di); taus,w=s._material(tr,b,x,dt,t); p,h=s._loads(di,b); C,R=s._basis(taus,p,h,xd,dt,t); g=R[:,:2]*taus; y=di['strain_curves_obs']-di['strain_0'][:,None]; r=di['instant_rate_obs']; Y=np.c_[y,r]; B=np.vstack((np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1])); n=len(h); lab=di['lab_id']; drift=R[:,2:]@w[2:]; one=np.ones(n); g0,g1=g.T; z0,z1=(h[:,None]-g).T
from scipy.optimize import least_squares
import json
P=np.vstack((np.c_[-np.expm1(-t[:,None]/taus)],1/taus)); A=Y@np.linalg.pinv(P.T); delta=A-h[:,None]*w[:2]; _,ss,V=np.linalg.svd(delta,full_matrices=False); print('fixed hold anchor delta singular values/direction',ss,V[0]); print('amplitude residual vs nominal drift',np.round(np.c_[lab,drift,delta],4))
for entry in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 q=np.array(entry['facility_telemetry_trace']); print('log',entry['completed_campaign_short_name'],'mean',q.mean(0).round(3),'sd',q.std(0).round(3))
cal=d['calibration_load_cells']; print('calibration fields',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in cal.items()]); print('total linear calibration residual RMSE',np.std(cal['bridge_raw']@b[1:]+b[0]-cal['true_channel_loads'].sum(1)))
sources={'tr+di':xd,'tr+te+di':s._features({k:np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')},di)[1], 'di_only':s._features({k:di[k][:0] for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')},di)[1]}
for source,xx in sources.items():
 v=w[2]+xx@w[3:]; print('feature source:',source)
 for kind in ['linear','affine','tanh','shift_tanh','rational','shift_rational','linear_tau','shift_tanh_tau']:
  shift=kind.startswith('shift') or kind=='affine'; change_tau=kind.endswith('_tau'); nonlinear='tanh' in kind or 'rational' in kind
  def pred(p):
   k=2; u=v.copy()
   if shift: u=u+p[k]; k+=1
   if nonlinear:
    a=np.exp(p[k]); k+=1
    u=np.tanh(a*u)/a if 'tanh' in kind else u/(1+a*u*u)
   ts=taus*np.exp(p[k:k+2]) if change_tau else taus
   M=np.vstack((-np.expm1(-t[:,None]/ts),1/ts)); return (h[:,None]*w[:2]+u[:,None]*p[:2])@M.T
  start=[.4,2.1]+([0.] if shift else [])+([-1.] if nonlinear else [])+([0.,0.] if change_tau else []); E=np.empty_like(Y); worst=[]
  for l in np.unique(lab):
   m=lab!=l; fit=least_squares(lambda p:(pred(p)-Y)[m].ravel(),start,max_nfev=150); E[~m]=(pred(fit.x)-Y)[~m]; worst.append(np.sqrt(np.mean(E[~m]**2,axis=0)).max())
  fit=least_squares(lambda p:(pred(p)-Y).ravel(),start,max_nfev=150); print(kind,np.round([np.std(fit.fun),np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),max(worst)],5),np.round(fit.x,5))
