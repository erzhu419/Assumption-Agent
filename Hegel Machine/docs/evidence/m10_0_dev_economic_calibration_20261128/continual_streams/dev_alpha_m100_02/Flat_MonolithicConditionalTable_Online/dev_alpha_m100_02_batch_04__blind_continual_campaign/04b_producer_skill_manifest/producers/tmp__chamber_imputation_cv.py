import pickle,sys,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_alpha_m100_02_batch_03_solver as s
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,g=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; c=s._bridge(d['calibration_load_cells']); xt,xe=s._features(tr,te); ta,th=s._fit(s._prepare(tr,xt,c),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; E=-np.expm1(-t[:,None]/ta); B=np.c_[E,t]; br=np.r_[1/ta,1.]; Z=np.c_[y,r]; C=np.linalg.lstsq(np.vstack([B,br]),Z.T,rcond=None)[0].T
v={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; ph,h,_=s._prepare(g,np.zeros((len(y),4)),c); hc=s._loads(g['hold_channel_raw'],c); pc=s._loads(g['past_channel_raw'],c); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0]; mem=ph@(np.exp(-tail[:,None]/ta)*-np.expm1(-dt[:,None]/ta)); base=np.c_[h[:,None]*th[:2],np.zeros(len(y))]; rem=C-base; ones=np.ones(len(y)); lab=g['lab_id']; print('taus weights',ta,th); print('residual coefficient statistics',np.quantile(rem,[0,.25,.5,.75,1],axis=0)); print('missing by col',np.isnan(g['coupon_features_raw']).sum(0)); print('hold hist range diag/test',np.min(h),np.max(h),np.min(mem,0),np.max(mem,0),np.quantile(s._loads(te['hold_channel_raw'],c).sum(-1),[0,.5,1]));
ref=v['ref_coupons']; raw=g['coupon_features_raw']; aligned=raw.copy();
for l in np.unique(lab):
 m=v['lab_id']==l; aligned[lab==l]=(raw[lab==l]-np.median(ref[m].mean(0)))/np.median(ref[m].std(0));
val,cell,n=np.unique(np.c_[v['lab_id'],v['stratum']],axis=0,return_inverse=True,return_counts=True); X=np.concatenate([xt,xe]); X0=v['coupon_features_raw']; w=1/n[cell.ravel()]; mean=(X*w[:,None]*(~np.isnan(X0))).sum(0)/((~np.isnan(X0))*w[:,None]).sum(0); xpool=np.where(np.isnan(aligned),mean,aligned); xboth=s._features(v,g)[1]; xsolo=s._features(g,g)[0];
from scipy.optimize import least_squares
complete=~np.isnan(raw).any(1)
Dmodels={}
for name,fill in [('weighted',xsolo[np.isnan(raw)]),('unweighted',np.broadcast_to(np.nanmean(aligned,axis=0),raw.shape)[np.isnan(raw)])]:
 x=aligned.copy(); x[np.isnan(x)]=fill; Dmodels[name]=th[2]+x@th[3:]
obs=np.c_[y-base@B.T,r-base@br]
G=np.vstack([B*np.r_[ta,1],np.ones(3)])
def model(d,f): return d[:,None]*np.r_[f,1-sum(f)]@G.T
for form,D in Dmodels.items():
 e=np.zeros_like(obs); fits=[]
 for j in np.unique(lab):
  m=lab==j; fit=(~m)&complete
  f=least_squares(lambda f:(model(D[fit],f)[:,:-1]-obs[fit,:-1]).ravel(), [.20,.52]).x
  fits.append(f); e[m]=model(D[m],f)-obs[m]
 print('diagnostic',form, 'fold fractions',np.round(fits,4), 'complete/missing strain and rate RMSE', [np.round([np.sqrt(np.mean(e[m,:-1]**2)),np.sqrt(np.mean(e[m,-1]**2))],6) for m in (complete,~complete)],'worst lab',np.max([np.sqrt(np.mean(e[lab==j]**2,axis=0)) for j in set(lab)]))
# Contrast fills for training on its own mechanism with intrinsic parameters fit only to complete coupons.
a=train=tr; q=~np.isnan(tr['coupon_features_raw']).any(axis=1)
ph,hh,ff=s._prepare(tr,xt,c); ti,wi=s._fit((ph[q],hh[q],ff[q]),dt,t,(tr['strain_curves_obs']-tr['strain_0'][:,None])[q],tr['instant_rate_obs'][q]); print('complete-only intrinsic',ti,wi)
for name,x in [('weighted',np.r_[xt,xe]),('unweighted',np.where(np.isnan(v['coupon_features_raw']),np.nanmean(np.where(np.isnan(v['coupon_features_raw']),np.nan,np.r_[xt,xe]),0),np.r_[xt,xe]))]:
 C,R=s._design(np.log(ti),s._prepare(tr,x[:len(q)],c),dt,t); er=R@wi-tr['instant_rate_obs']; ec=C@wi-(tr['strain_curves_obs']-tr['strain_0'][:,None]); print('train',name,'held-out missing median absolute strain/rate',np.median(abs(ec[~q])),np.median(abs(er[~q])))
