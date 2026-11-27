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
from scipy.optimize import minimize_scalar
from scipy.special import expit
lab=dg['lab_id']; raw=dg['coupon_features_raw']; complete=np.isfinite(raw).all(1)
opt=minimize_scalar(lambda k:np.sum((target-(target@Z(k)/(Z(k)@Z(k)))[:,None]*Z(k))**2),bounds=(.8,1.2),method='bounded'); q=Z(opt.x); amp=target@q/(q@q)
print('time',opt.x,'shape RMS',np.sqrt(np.mean((target-amp[:,None]*q)**2)))
print('lab x1..4 drift amp err last z0 z1'); dr=theta[2]+xnew@theta[3:]; print(np.round(np.c_[lab,complete,xnew,dr,amp,amp-dr,p[:,-1],z],3))
for xx in [xnew,xd]:
 dr=theta[2]+xx@theta[3:]; err=amp-dr
 specs={'zero':np.empty((n,0)),'affine':np.c_[np.ones(n),dr],'last/origin':np.c_[dr,p[:,-1]-p[:,-1].mean()],'last':np.c_[np.ones(n),dr,p[:,-1]],'last*drift':np.c_[np.ones(n),dr,p[:,-1],p[:,-1]*dr],'hold':np.c_[np.ones(n),dr,h],'mem0':np.c_[np.ones(n),dr,z[:,0]],'history ridge':np.c_[np.ones(n),dr,p],'sigmoid':np.c_[np.ones(n),dr,expit(p[:,-1])],'features':np.c_[np.ones(n),xx], 'nonlinear drift':np.c_[np.ones(n),dr,dr**2], 'features sqrt':np.c_[np.ones(n),np.sign(xx)*np.sqrt(abs(xx))], 'features signed-cubic':np.c_[np.ones(n),xx,xx**3]}
 for key,F in specs.items():
  for reg in ([.2,1,3] if key=='history ridge' else [0.]):
   def fit(mask):return np.linalg.lstsq(np.r_[F[mask],np.sqrt(reg)*np.eye(F.shape[1])],np.r_[err[mask],np.zeros(F.shape[1])],rcond=None)[0]
   def cv(groups):
    out=err.copy()
    for g in np.unique(groups):
     test=groups==g; out[test]-=F[test]@fit(~test)
    return [np.sqrt(np.mean(out*out)),max(np.sqrt(np.mean(out[groups==g]**2)) for g in np.unique(groups))]
   c=fit(np.ones(n,bool)); print(key,reg,'fit/lab RMS,worst/coupon RMS,max',np.round([np.sqrt(np.mean((err-F@c)**2)),*cv(lab),*cv(np.arange(n))],4),'coef',c.round(4))
for key in ('past_channel_raw','hold_channel_raw','aux_chamber_telemetry'):
 for block in (tr,te,dg):
  if key in block:
   a=np.asarray(block[key]); print('distribution',key,len(a),a.shape,np.nanmean(a.reshape(len(a),-1),axis=0).round(3)[:24],np.nanstd(a.reshape(len(a),-1),axis=0).round(3)[:24])
