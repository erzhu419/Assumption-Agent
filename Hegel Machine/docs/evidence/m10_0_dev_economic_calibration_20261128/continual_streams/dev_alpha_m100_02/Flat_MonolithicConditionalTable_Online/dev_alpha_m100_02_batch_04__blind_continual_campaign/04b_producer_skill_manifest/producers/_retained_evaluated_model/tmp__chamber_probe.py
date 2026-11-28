import pickle,sys,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_alpha_m100_02_batch_03_solver as s
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,g=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; c=s._bridge(d['calibration_load_cells']); xt,xe=s._features(tr,te); ta,th=s._fit(s._prepare(tr,xt,c),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; E=-np.expm1(-t[:,None]/ta); B=np.c_[E,t]; br=np.r_[1/ta,1.]; Z=np.c_[y,r]; C=np.linalg.lstsq(np.vstack([B,br]),Z.T,rcond=None)[0].T
v={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; ph,h,_=s._prepare(g,np.zeros((len(y),4)),c); hc=s._loads(g['hold_channel_raw'],c); pc=s._loads(g['past_channel_raw'],c); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0]; mem=ph@(np.exp(-tail[:,None]/ta)*-np.expm1(-dt[:,None]/ta)); base=np.c_[h[:,None]*th[:2],np.zeros(len(y))]; rem=C-base; ones=np.ones(len(y)); lab=g['lab_id']; print('taus weights',ta,th); print('residual coefficient statistics',np.quantile(rem,[0,.25,.5,.75,1],axis=0)); print('missing by col',np.isnan(g['coupon_features_raw']).sum(0)); print('hold hist range diag/test',np.min(h),np.max(h),np.min(mem,0),np.max(mem,0),np.quantile(s._loads(te['hold_channel_raw'],c).sum(-1),[0,.5,1]));
ref=v['ref_coupons']; raw=g['coupon_features_raw']; aligned=raw.copy();
for l in np.unique(lab):
 m=v['lab_id']==l; aligned[lab==l]=(raw[lab==l]-np.median(ref[m].mean(0)))/np.median(ref[m].std(0));
val,cell,n=np.unique(np.c_[v['lab_id'],v['stratum']],axis=0,return_inverse=True,return_counts=True); X=np.concatenate([xt,xe]); X0=v['coupon_features_raw']; w=1/n[cell.ravel()]; mean=(X*w[:,None]*(~np.isnan(X0))).sum(0)/((~np.isnan(X0))*w[:,None]).sum(0); xpool=np.where(np.isnan(aligned),mean,aligned); xboth=s._features(v,g)[1]; xsolo=s._features(g,g)[0];
for name,x in [('pool',xpool),('joint',xboth),('solo',xsolo)]:
 drift=th[2]+x@th[3:]; print(name,'drift range',np.quantile(drift,[0,.5,1]),'initial drift max residual',np.max(abs(rem@br-drift))); print('drift vs coeff correlated',np.corrcoef(np.c_[drift,rem].T)[0,1:]); print('drift mismatch rows',np.c_[r-(base@br+drift),np.isnan(raw).any(1)][np.argsort(abs(r-(base@br+drift)))[-6:]]);
 def score(pred):
  e=pred@B.T-y; er=pred@br-r; return np.sqrt(np.mean(e**2)),max(np.sqrt(np.mean(e[lab==l]**2)) for l in set(lab)),np.sqrt(np.mean(er**2)),np.max(abs(er))
 def fit(F):
  pred=np.zeros_like(C)
  for i in range(len(y)):
   m=np.arange(len(y))!=i; pred[i]=base[i]+F[i]@np.linalg.lstsq(F[m],rem[m],rcond=None)[0];
  return np.array(score(pred)).round(5)
 print('archive reset',score(base+np.c_[drift[:,None]*ta*s._RESET_DRIFT_FRACTIONS,drift*(1-s._RESET_DRIFT_FRACTIONS.sum())]));
 for key,F in {'drift':drift[:,None],'intercept drift':np.c_[ones,drift],'drift polynomial':np.c_[ones,drift,drift**2,drift**3], 'hold drift':np.c_[ones,h,drift], 'hold drift interact':np.c_[ones,h,drift,h*drift,h**2,drift**2],'history drift':np.c_[ones,mem,drift], 'history hold drift':np.c_[ones,h,mem,drift], 'features':np.c_[ones,x],'features hold':np.c_[ones,x,h], 'features hold history':np.c_[ones,x,h,mem], 'features all hold channels':np.c_[ones,x,hc],'incomplete flags':np.c_[ones,drift,np.isnan(raw)]}.items(): print(key,fit(F));
 if name=='pool': print('rows: hold, drift, natural scaled branch residues, steady, rate excess, max abs past, features'); print(np.c_[h,drift,rem[:,:2]/ta,rem[:,2],r-base@br-drift,abs(ph).max(1),x].round(3));
