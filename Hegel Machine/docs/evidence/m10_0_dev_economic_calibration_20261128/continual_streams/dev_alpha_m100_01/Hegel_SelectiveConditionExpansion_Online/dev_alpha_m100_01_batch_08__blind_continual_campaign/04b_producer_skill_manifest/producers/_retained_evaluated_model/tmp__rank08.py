import sys,pickle,numpy as np
sys.path[:0]=['/holdout_task','/tmp/hegel_continual_archive']
import _verified_upstream as up
import dev_alpha_m100_01_batch_02_solver as old
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=up.calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
a,b,c=[d[k] for k in ('train','test','chamber_verification_trials')]; t,dt=d['eval_times'],d['past_dt']; n=len(a['lab_id']); x=old._features(a,b); xc=old._features(a,c)[n:]; p,h=old._loads(c)
tau,th=old._fit(*old._loads(a),x[:n],dt,t,a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']);print('tau theta',tau,th)
Y=c['strain_curves_obs']-c['strain_0'][:,None]; R=c['instant_rate_obs']; K=-np.expm1(-t[:,None]/tau); B=np.c_[K,t]; s=np.r_[1/tau,1.]; cc,rr=old._design(tau,p,h,xc,dt,t); base=cc@th; rbase=rr@th; dr=rr[:,2:]@th[2:]; g=rr[:,:2]*tau; print('baseline curve,rate RMSE',np.sqrt(np.mean((base-Y)**2)),np.sqrt(np.mean((rbase-R)**2)));print('diag missing',np.isnan(c['coupon_features_raw']).sum(0));print('rate comparison: observed,base,reset,drift',np.round(np.c_[R,rbase,h*(th[:2]/tau).sum()+dr,dr],4)); print('curve singular values',np.linalg.svd(Y,compute_uv=False))
def robust(A,y,scale):
 return least_squares(lambda w:A@w-y,np.linalg.lstsq(A,y,rcond=None)[0],loss='cauchy',f_scale=scale).x
U=h[:,None]*th[:2]; D=Y-U@K.T; S=R-U@(1/tau)
B0=np.c_[tau*K,t]; s0=np.ones(3)
print('lab/strata',[(np.unique(z['lab_id'],return_counts=True),np.unique(z['stratum'],return_counts=True)) for z in (a,b,c)])
def fit_response(idx):
    def residual(v):
        frac=np.r_[v[:2],1-v[:2].sum()]; amp=v[2:]; return np.r_[((amp[:,None]*(B0@frac)-D[idx])/.002).ravel(), (amp-S[idx])/.008]
    return least_squares(residual,np.r_[.23,.59,S[idx]],loss='soft_l1',max_nfev=75).x[:2]
def fit_score(idx,frac,score):
    profile=B0@np.r_[frac,1-frac.sum()]; y=np.c_[D,S]; f=np.r_[profile,1.]; sig=np.r_[np.full(len(t),.002),.008]; w=f/sig**2; amp=y@w/(f@w)
    p=least_squares(lambda v: (score[idx]@v-amp[idx])/.013,np.linalg.lstsq(score[idx],amp[idx],rcond=None)[0],loss='soft_l1').x
    return profile,p,amp
nc=len(h); missing=np.any(np.isnan(c['coupon_features_raw']),axis=1)
folds=[('LOO '+str(i),np.arange(nc)==i) for i in range(nc)]+[('lab '+str(i),c['lab_id']==i) for i in np.unique(c['lab_id'])]+[('missing',missing),('low drift',dr<np.quantile(dr,.2)),('high drift',dr>np.quantile(dr,.8))]
fx=old._features(a,dict(c,**{k:np.concatenate((b[k],c[k]),axis=0) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}))[n+len(b['lab_id']):];
print('diagnostic drift conventions: max score errors vs amplitude')
for label,xx in [('train+diag',xc),('all',fx)]:
  frac=fit_response(np.arange(nc)); profile,p,amp=fit_score(np.arange(nc),frac,np.c_[np.ones(nc),xx]); raw=th[2]+xx@th[3:]; print(label,'fractions',np.r_[frac,1-frac.sum()],'feature coefficients',p,'raw drift vs fitted amp',np.quantile(abs(amp-raw),[.5,.9,1]),'exact time basis residual',np.quantile(abs(D-amp[:,None]*profile),[.5,.9,1])); print('amp/drift pairs',np.round(np.c_[raw,amp,amp-raw,missing],4).tolist())
  scores={'intrinsic':raw[:,None],'affine':np.c_[np.ones(nc),raw],'quadratic':np.c_[np.ones(nc),raw,raw**2],'full_features':np.c_[np.ones(nc),xx]}
  for kind,F in scores.items():
    es=[]; rr=[]; slices=[]
    for j,(name,m) in enumerate(folds):
      q,coef,_=fit_score(np.where(~m)[0],fit_response(np.where(~m)[0]),F); A=F[m]@coef; E=abs(A[:,None]*q-D[m]); R0=abs(A-S[m]);
      if j<nc: es.extend(E.ravel()); rr.extend(R0)
      else: slices.append((name,round(np.sqrt(np.mean(E**2)),4),round(np.sqrt(np.mean(R0**2)),4)))
    print(kind,'LOO p90/max curve',np.quantile(es,[.9,1]),'rate',np.quantile(rr,[.9,1]),'blocked RMSE',slices)
