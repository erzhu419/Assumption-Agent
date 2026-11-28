import pickle,sys,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_alpha_m100_02_batch_03_solver as s
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,g=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; c=s._bridge(d['calibration_load_cells']); xt,xe=s._features(tr,te); ta,th=s._fit(s._prepare(tr,xt,c),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; E=-np.expm1(-t[:,None]/ta); B=np.c_[E,t]; br=np.r_[1/ta,1.]; Z=np.c_[y,r]; C=np.linalg.lstsq(np.vstack([B,br]),Z.T,rcond=None)[0].T
v={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; ph,h,_=s._prepare(g,np.zeros((len(y),4)),c); hc=s._loads(g['hold_channel_raw'],c); pc=s._loads(g['past_channel_raw'],c); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0]; mem=ph@(np.exp(-tail[:,None]/ta)*-np.expm1(-dt[:,None]/ta)); base=np.c_[h[:,None]*th[:2],np.zeros(len(y))]; rem=C-base; ones=np.ones(len(y)); lab=g['lab_id']; print('taus weights',ta,th); print('residual coefficient statistics',np.quantile(rem,[0,.25,.5,.75,1],axis=0)); print('missing by col',np.isnan(g['coupon_features_raw']).sum(0)); print('hold hist range diag/test',np.min(h),np.max(h),np.min(mem,0),np.max(mem,0),np.quantile(s._loads(te['hold_channel_raw'],c).sum(-1),[0,.5,1]));
ref=v['ref_coupons']; raw=g['coupon_features_raw']; aligned=raw.copy();
for l in np.unique(lab):
 m=v['lab_id']==l; aligned[lab==l]=(raw[lab==l]-np.median(ref[m].mean(0)))/np.median(ref[m].std(0));
val,cell,n=np.unique(np.c_[v['lab_id'],v['stratum']],axis=0,return_inverse=True,return_counts=True); X=np.concatenate([xt,xe]); X0=v['coupon_features_raw']; w=1/n[cell.ravel()]; mean=(X*w[:,None]*(~np.isnan(X0))).sum(0)/((~np.isnan(X0))*w[:,None]).sum(0); xpool=np.where(np.isnan(aligned),mean,aligned); xboth=s._features(v,g)[1]; xsolo=s._features(g,g)[0];
mask=np.isnan(raw)
D=th[2]+xsolo@th[3:]; K=np.c_[B*np.r_[ta,1.]].T
# Get a drift estimate using all observations, without using any feature values.
F=np.array([.1958,.525,.2792]); response=np.r_[F@K,1.]; target=np.c_[y-base@B.T,r-base@br]; Dobs=target@response/(response@response)
print('col, local imputation, inferred from initial rate, inferred from trajectory, train/test pooled fill, joint fill')
for i,j in zip(*np.where(mask)):
 print(j,np.round([xsolo[i,j],xsolo[i,j]+(r[i]-base[i]@br-D[i])/th[3+j],xsolo[i,j]+(Dobs[i]-D[i])/th[3+j],xpool[i,j],xboth[i,j]],4))
for name,part in [('train',tr),('test',te),('diag',g)]:
 ref=part['ref_coupons']; lab=part['lab_id']; raw=part['coupon_features_raw']; al=raw.copy()
 for l in set(lab):
  z=ref[lab==l];al[lab==l]=(raw[lab==l]-np.median(z.mean(0)))/np.median(z.std(0))
 _,idx,n=np.unique(np.c_[lab,part['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[idx.ravel()]; A=np.isfinite(al)
 print(name,'means unweighted, inverse count observed normalized, inverse count global denominator, zero/local ref, and fills from solver:')
 print(np.array([np.nanmean(al,axis=0),np.nansum(al*w[:,None],axis=0)/(w@A),np.nansum(al*w[:,None],axis=0)/w.sum(),np.array([s._features(part,part)[0][np.isnan(raw[:,j]),j][0] if np.isnan(raw[:,j]).any() else np.nan for j in range(4)])]).round(5))
