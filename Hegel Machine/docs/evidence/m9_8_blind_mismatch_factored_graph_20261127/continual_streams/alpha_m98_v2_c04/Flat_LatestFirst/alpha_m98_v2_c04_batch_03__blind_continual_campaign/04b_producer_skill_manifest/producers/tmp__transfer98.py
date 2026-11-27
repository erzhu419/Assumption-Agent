import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c04_s00_solver as s
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; dg=item['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; br=d['calibration_load_cells']; raw=br['bridge_raw']; true=br['true_channel_loads']; b=s._bridge(br)
xb=s._features(tr,te); x=xb[:len(tr['lab_id'])]; p,h=s._loads(tr,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,theta=s._fit(x,p,h,dt,t,y,r)
print('tau',tau,'theta',theta); print('linear bridge rms, max by channel',np.sqrt(np.mean((np.c_[np.ones(len(raw)),raw]@b-true)**2,0)),np.max(np.abs(np.c_[np.ones(len(raw)),raw]@b-true),0))
for k in br['single_channel_sweeps']: print('sweep',k,np.shape(br['single_channel_sweeps'][k]),np.asarray(br['single_channel_sweeps'][k]).ravel()[:10])
for name in ['ref_coupons','lab_id','stratum','coupon_features_raw']:
 pass
joint={k:np.concatenate([te[k],dg[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}; xd=s._features(tr,joint)[-len(dg['lab_id']):]
from scipy.optimize import minimize_scalar
p,h=s._loads(dg,b); Y=dg['strain_curves_obs']-dg['strain_0'][:,None]; R=dg['instant_rate_obs']; A=np.c_[-np.expm1(-t[:,None]/tau),t]; B=np.c_[np.r_[A[:,0],1/tau[0]],np.r_[A[:,1],1/tau[1]],np.r_[t,1.]]; obs=np.c_[Y,R];
lab=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xnew=dg['coupon_features_raw'].copy(); st=np.r_[tr['stratum'],te['stratum']]; counts=np.unique(np.c_[lab,st],axis=0,return_inverse=True,return_counts=True); w=1./counts[2][counts[1],None]; rawx=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); fill=np.sum(w*xb*(~np.isnan(rawx)),0)/np.sum(w*(~np.isnan(rawx)),0)
for g in np.unique(lab):
 panel=refs[lab==g]; xnew[dg['lab_id']==g]=(xnew[dg['lab_id']==g]-np.median(panel.mean(0)))/np.median(panel.std(0))
xnew=np.where(np.isnan(xnew),fill,xnew)
print('max diag feature coordinate difference',np.max(abs(xnew-xd),0))
p,h=s._loads(dg,b); n=len(h); later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; m=-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau); z=p@m; gap=h[:,None]-z; print('dt, t',dt,t); print('diag keys',list(dg)); print('raw and calibrated hold ranges',h.min(),h.max())
D=B[:,:2]; target=obs-(gap*theta[:2])@D.T
Z=lambda k:np.r_[-k*np.expm1(-t/k),1.]
import json
L=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for entry in L:
 T=np.asarray(entry['facility_telemetry_trace']); print('apparatus', {k:v for k,v in entry.items() if k!='facility_telemetry_trace'},T.shape,T.mean(0).round(3),T.std(0).round(3))
print('dataset keys',list(d),'test keys',list(te))
for block in [tr,te,dg]:
 print('panel',len(block['lab_id']),np.shape(block['ref_coupons']),'missing features',np.isnan(block['coupon_features_raw']).sum(0))
 for g in np.unique(block['lab_id']):
  panel=block['ref_coupons'][block['lab_id']==g]; print(g,len(panel),np.median(panel.mean(0)),np.median(panel.std(0)))
grps=dg['lab_id']; lab=np.r_[tr['lab_id'],te['lab_id']]; combined={k:np.concatenate([te[k],dg[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}
full={k:np.concatenate([tr[k],combined[k]]) for k in combined}; features={'train+test':xnew,'all':xd}
for source,block in [('train',tr),('test',te),('diag',dg)]:
 xx=dg['coupon_features_raw'].copy()
 for g in np.unique(grps):
  panel=block['ref_coupons'][block['lab_id']==g]; center=np.median(panel.mean(0)); scale=np.median(panel.std(0)); ii=grps==g
  xx[ii]=(xx[ii]-center)/scale if scale>1e-8 else xnew[ii]
 features[source]=np.where(np.isnan(xx),fill,xx)
print('diag missing locations',np.argwhere(np.isnan(dg['coupon_features_raw'])).tolist())
# Profile shared chamber terms (not coupon-specific amplitudes); analytically include their initial derivatives.
from scipy.optimize import minimize_scalar
base=(gap*theta[:2])@D.T; ones=np.ones(n)
def mat(F,q):return (F[:,None,:]*q[None,:,None]).reshape(n*len(q),F.shape[1])
def evaluate(xx,mem,driftkind,kind='exp'):
 drift=theta[2]+xx@theta[3:]; df={'fixed':None,'affine':np.c_[ones,drift],'feature':np.c_[ones,xx]}[driftkind]
 ff={'none':[],'gap':[(gap[:,[j]],D[:,j]) for j in range(2)],'own':[(np.c_[h,z[:,j]],D[:,j]) for j in range(2)],'cross':[(np.c_[h,z],D[:,j]) for j in range(2)],'bias':[(ones[:,None],D[:,j]) for j in range(2)]}[mem]
 def fit(mask,k,ret=False):
  q={'exp':lambda:-k*np.expm1(-t/k),'rational':lambda:t/(1+t/k),'tanh':lambda:k*np.tanh(t/k)}[kind](); q=np.r_[q,1.]; offset=base.copy(); fs=[mat(F,v) for F,v in ff]
  if df is None: offset+=drift[:,None]*q
  else: fs.append(mat(df,q))
  iy=np.repeat(mask,obs.shape[1]); y=(obs-offset).ravel(); X=np.hstack(fs) if fs else np.empty((len(y),0)); co=np.linalg.lstsq(X[iy],y[iy],rcond=None)[0]; e=(y-X@co).reshape(obs.shape)
  return (e,co) if ret else np.mean(e[mask]**2)
 def best(mask):return minimize_scalar(lambda k:fit(mask,k),bounds=(.35,3.5),method='bounded').x
 allrows=np.ones(n,dtype=bool); k=best(allrows); e,co=fit(allrows,k,True); cv=e.copy(); kc=[]
 for g in np.unique(grps):
  mask=grps!=g; k1=best(mask); kc.append(k1); cv[~mask]=fit(mask,k1,True)[0][~mask]
 return np.sqrt(np.mean(e**2)),max(np.sqrt(np.mean(cv[grps==g]**2)) for g in np.unique(grps)),np.percentile(abs(cv[:,:-1]),90),np.percentile(abs(cv[:,-1]),90),k,kc,co
for source,xx in features.items():
 for mem in ['none','bias','gap','own','cross']:
  for driftkind in ['fixed','affine','feature']:
   if source!='all' and mem in ['bias','cross']: continue
   r=evaluate(xx,mem,driftkind); print('fit',source,mem,driftkind,'rms/max-lab/p90curve/p90rate/k:',np.round(r[:5],5),'coef',np.round(r[-1],3))
for kind in ['rational','tanh']:
 r=evaluate(xd,'own','fixed',kind); print('alternative',kind,np.round(r[:5],5))
