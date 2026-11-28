import pickle,sys,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_alpha_m100_02_batch_03_solver as s
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,g=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; c=s._bridge(d['calibration_load_cells']); xt,xe=s._features(tr,te); ta,th=s._fit(s._prepare(tr,xt,c),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; E=-np.expm1(-t[:,None]/ta); B=np.c_[E,t]; br=np.r_[1/ta,1.]; Z=np.c_[y,r]; C=np.linalg.lstsq(np.vstack([B,br]),Z.T,rcond=None)[0].T
v={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; ph,h,_=s._prepare(g,np.zeros((len(y),4)),c); hc=s._loads(g['hold_channel_raw'],c); pc=s._loads(g['past_channel_raw'],c); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0]; mem=ph@(np.exp(-tail[:,None]/ta)*-np.expm1(-dt[:,None]/ta)); base=np.c_[h[:,None]*th[:2],np.zeros(len(y))]; rem=C-base; ones=np.ones(len(y)); lab=g['lab_id']; print('taus weights',ta,th); print('residual coefficient statistics',np.quantile(rem,[0,.25,.5,.75,1],axis=0)); print('missing by col',np.isnan(g['coupon_features_raw']).sum(0)); print('hold hist range diag/test',np.min(h),np.max(h),np.min(mem,0),np.max(mem,0),np.quantile(s._loads(te['hold_channel_raw'],c).sum(-1),[0,.5,1]));
ref=v['ref_coupons']; raw=g['coupon_features_raw']; aligned=raw.copy();
for l in np.unique(lab):
 m=v['lab_id']==l; aligned[lab==l]=(raw[lab==l]-np.median(ref[m].mean(0)))/np.median(ref[m].std(0));
val,cell,n=np.unique(np.c_[v['lab_id'],v['stratum']],axis=0,return_inverse=True,return_counts=True); X=np.concatenate([xt,xe]); X0=v['coupon_features_raw']; w=1/n[cell.ravel()]; mean=(X*w[:,None]*(~np.isnan(X0))).sum(0)/((~np.isnan(X0))*w[:,None]).sum(0); xpool=np.where(np.isnan(aligned),mean,aligned); xboth=s._features(v,g)[1]; xsolo=s._features(g,g)[0];
fobs=~np.isnan(raw).any(1)
dobs=r-base@br
print('dsolo,dfromrate,amplitude residual in rate units,missing mask,lab')
print(np.c_[th[2]+xsolo@th[3:],dobs,rem[:,:2]/ta,rem[:,2],np.isnan(raw).astype(int),lab].round(4))
for name,z in [('train',tr),('test',te),('diag',g)]:
 print(name,'alignment by lab: n, center, scale; raw missing',np.isnan(z['coupon_features_raw']).sum(0))
 for l in np.unique(z['lab_id']):
  a=z['ref_coupons'][z['lab_id']==l]; print(l,len(a),np.median(a.mean(0)),np.median(a.std(0)))
 if name=='test': print('test drift joint vs solo ranges',np.quantile(np.c_[th[2]+xe@th[3:],th[2]+s._features(te,te)[0]@th[3:]],[0,.25,.5,.75,1],axis=0))
from scipy.optimize import least_squares
D=th[2]+xsolo@th[3:]; M=rem/np.r_[ta,1];
for ds in [D,dobs]:
 for name,model,p0 in [('linear',lambda p,d: d*p[0], [0.4]),('even-rational',lambda p,d: p[0]*d/(1+p[1]*d*d),[.4,.1]),('cubic',lambda p,d: p[0]*d+p[1]*d*d+p[2]*d**3,[.4,0,0]),('offset',lambda p,d: p[0]*d+p[1],[.4,0]),('saturation',lambda p,d:p[0]*np.tanh(p[1]*d),[.4,1])]:
  fit=[]; rr=[]
  for k in range(3):
   p=least_squares(lambda p:model(p,ds[fobs])-M[fobs,k],p0).x; fit.append(p.round(4));rr.append(np.sqrt(np.mean((model(p,ds[fobs])-M[fobs,k])**2)))
  print('law',name,fit,np.round(rr,5))
