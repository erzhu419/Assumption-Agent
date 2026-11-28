import pickle,numpy as np
from scipy.optimize import minimize_scalar
from m100_base import _bridge,_loads,_features,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=170)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=_features(tr,te); xd=_features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)[-len(q['lab_id']):]; b=_bridge(d['calibration_load_cells']); tau,theta=_fit(*_loads(tr,b),x[:n],dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); p,h=_loads(q,b); C,R=_design(tau,p,h,xd,dt,t); ec=q['strain_curves_obs']-q['strain_0'][:,None]-C@theta; er=q['instant_rate_obs']-R@theta; E=np.c_[er,ec]
def temporal(k,rate=True):
 L=np.c_[-np.expm1(-t/k),t]; return np.vstack(([1/k,1],L)) if rate else L
def err(k,rate=True):
 L=temporal(k,rate); Y=E if rate else ec; return Y-(Y@np.linalg.pinv(L.T))@L.T
from scipy.optimize import least_squares
D=np.c_[q['instant_rate_obs'],q['strain_curves_obs']-q['strain_0'][:,None]]; T=np.r_[1.,t]; K=np.array([np.r_[1/k,-np.expm1(-t/k)] for k in tau]); raw=q['coupon_features_raw']; lab=q['lab_id']; O=np.eye(4)[lab]; valid=~np.isnan(raw).any(1); U=np.nan_to_num(raw)@theta[3:]; keys=('lab_id','stratum','coupon_features_raw','ref_coupons'); merge=lambda *a:{k:np.concatenate([v[k] for v in a]) for k in keys}
H=theta[:2]@K; W=D-h[:,None]*H
for name,ref in [('train',tr),('train+q',merge(tr,q)),('train+test',merge(tr,te)),('all',merge(tr,te,q)),('q',q),('test+q',merge(te,q))]:
 center=[]; scale=[]
 for j in range(4):
  rows=ref['ref_coupons'][ref['lab_id']==j]; center.append(np.median(rows.mean(0))); scale.append(np.median(rows.std(0)))
 X=(raw-np.array(center)[lab,None])/np.array(scale)[lab,None]; F=np.c_[np.ones(len(h)),np.nan_to_num(X)@theta[3:]]; B=np.linalg.lstsq(F[valid],W[valid],rcond=None)[0]; print(name,'complete RMS',np.mean((F@B-W)[valid]**2)**.5,'lab RMS',[(j,round(np.mean((F@B-W)[valid&(lab==j)]**2)**.5,5)) for j in range(4)])
 if name in ('train+q','train+test'):
  V=_features(tr,q)[-len(h):] if name=='train+q' else np.where(np.isnan(X),np.average(x,axis=0),X); score=(W-B[0])@B[1]/(B[1]@B[1]); ii,jj=np.nonzero(np.isnan(raw)); print('missing feature, implied standardized, mean imputation',np.c_[jj,(score[ii]-F[ii,1])/theta[3:][jj],V[ii,jj]].round(3)); print('B intercept slope',B.round(4))
F=np.c_[O,O*U[:,None]]; B=np.linalg.lstsq(F[valid],W[valid],rcond=None)[0]; print('unconstrained centered hold residual',np.mean((W-F@B)[valid]**2)**.5); _,s,v=np.linalg.svd(B,full_matrices=False); print('feature/intercept SVD',s.round(5),'projections', (B@v[0]).round(5)); print('laboratory center/scale relative to lab0 fitted',((B@v[0])[4:]/(B@v[0])[4]).round(4)); print('raw means across splits, range',[(np.nanmean(v['coupon_features_raw'],axis=0).round(3), np.sum(np.isnan(v['coupon_features_raw']),axis=0)) for v in (tr,te,q)])
