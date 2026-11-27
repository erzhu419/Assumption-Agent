import runpy,pickle,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_05_solver.py'); p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; a=p['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; cal=m['_stress_calibration'](d['calibration_load_cells'])
def prep(s,x):return (m['_channel_loads'](s['past_channel_raw'],cal).sum(-1),m['_channel_loads'](s['hold_channel_raw'],cal).sum(-1),x)
x=m['_aligned_features'](tr,te); n=len(tr['lab_id']); train=prep(tr,x[:n]); tau,th=m['_fit'](dt,t,train,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau, theta',tau,th)
def mad(x,axis=None):return 1.4826*np.median(abs(x-np.median(x,axis=axis,keepdims=True)),axis=axis)
c,r=m['_design'](tau,dt,t,train); e=c@th-(tr['strain_curves_obs']-tr['strain_0'][:,None]); f=r@th-tr['instant_rate_obs']; print('train residual mad',mad(e),mad(f),'by time',mad(e,0))
y=a['strain_curves_obs']-a['strain_0'][:,None]; yr=a['instant_rate_obs']; lab=a['lab_id']; g=-np.expm1(-t[:,None]/tau); u=g-t[:,None]/tau
for alignment in ('diagnostic-local','reference-joint'):
 if alignment=='diagnostic-local': xx=m['_aligned_features'](a,a)[:len(lab)]
 else:
  xx=a['coupon_features_raw'].copy(); all_lab=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); st=np.r_[tr['stratum'],te['stratum']]; raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]);
  for j in np.unique(lab):
   q=refs[all_lab==j]; center=np.nanmedian(np.nanmean(q,0)); scale=np.nanmedian(np.nanstd(q,0)); xx[lab==j]=(xx[lab==j]-center)/scale; raw[all_lab==j]=(raw[all_lab==j]-center)/scale
  _,ids,k=np.unique(np.c_[all_lab,st],axis=0,return_inverse=True,return_counts=True); w=1/k[ids,None]; fill=np.nansum(w*raw,0)/np.sum(w*np.isfinite(raw),0); xx=np.c_[np.ones(len(lab)),np.where(np.isnan(xx),fill,xx)]
 coupon=prep(a,xx); des,rate=m['_design'](tau,dt,t,coupon); drift=xx@th[2:]; hold=coupon[1]; z_term=hold[:,None]*th[:2]-rate[:,:2]*tau*th[:2]; base=(hold[:,None]*th[:2])@g.T+drift[:,None]*t; baser=(hold[:,None]*th[:2]/tau).sum(1)+drift
 mem=-z_term[:,None,:]*g; memr=-z_term/tau; forc=drift[:,None,None]*u; zero=np.zeros_like(memr); slope=drift[:,None]*t
 print('\n',alignment,'diagnostic raw ranges',y.min(),y.max(),yr.min(),yr.max())
 def evaluate(name,base,baser,C=None,R=None):
  if C is None:
   ec=base-y; er=baser-yr; pars=[]
  else:
   def fit(mask):
    def resid(b):return np.r_[((base[mask]+C[mask]@b-y[mask])/.018).ravel(),(baser[mask]+R[mask]@b-yr[mask])/.030]
    return least_squares(resid,np.zeros(C.shape[-1]),loss='soft_l1',f_scale=1).x
   ec=np.zeros_like(y); er=np.zeros_like(yr)
   for j in np.unique(lab):
    k=lab==j; b=fit(~k); ec[k]=base[k]+C[k]@b-y[k]; er[k]=baser[k]+R[k]@b-yr[k]
   pars=fit(np.ones(len(lab),bool)).round(4)
  print(name,pars,'med',np.median(abs(ec)).round(5),np.median(abs(er)).round(5),'worst lab med',max(np.median(abs(ec[lab==j])) for j in np.unique(lab)).round(5),max(np.median(abs(er[lab==j])) for j in np.unique(lab)).round(5),'90pct',np.quantile(abs(ec),.9).round(5),np.quantile(abs(er),.9).round(5))
 for alpha,label in ((0,'reset'),(1,'carry')):
  b=base+alpha*mem.sum(2); br=baser+alpha*memr.sum(1)
  evaluate(label,b,br); evaluate(label+' force',b,br,forc,zero); evaluate(label+' force+slope',b,br,np.dstack([forc,slope]),np.c_[zero,drift]);
 evaluate('shared carry + force',base,baser,np.dstack([mem.sum(2),forc]),np.c_[memr.sum(1),zero]); evaluate('branch carry + force',base,baser,np.dstack([mem,forc]),np.c_[memr,zero]); evaluate('branch carry + force+slope',base,baser,np.dstack([mem,forc,slope]),np.c_[memr,zero,drift]);
 for alpha,f in [(0,[.34880024,1.63895287]),(1,[.41228894,2.2768587])]:evaluate('archived '+str(alpha),base+alpha*mem.sum(2)+forc@f,baser+alpha*memr.sum(1))
