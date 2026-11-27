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
lab=dg['lab_id']; raw=dg['coupon_features_raw']; good=np.isfinite(raw).all(1); rr=theta[2]+xnew@theta[3:]; aa=theta[2]+xd@theta[3:]; trraw=tr['coupon_features_raw']; telab=te['lab_id']; trlab=tr['lab_id'];
def learn_cal(blocks):
 l=np.concatenate([v['lab_id'] for v in blocks]); a=np.vstack([v['ref_coupons'] for v in blocks]); return {g:(np.median(a[l==g].mean(0)),np.median(a[l==g].std(0))) for g in np.unique(l)}
print('panel calibrations',[(name,learn_cal(bs)) for name,bs in [('train',[tr]),('train+test',[tr,te]),('diag',[dg]),('all',[tr,te,dg])]])
base=lambda k:Z(k)
def profile(ip):
 def loss(k):
  q=base(k); return np.sum((target[ip]-(target[ip]@q/(q@q))[:,None]*q)**2)
 k=minimize_scalar(loss,bounds=(.8,1.2),method='bounded').x; q=base(k); return k,q,target@q/(q@q)
k,q,amp=profile(np.ones(n,bool)); print('time',k); print('lab complete x drift joint drift all amplitude past_last residual')
print(np.round(np.c_[lab,good,xnew,rr,aa,amp,p[:,-1],amp-rr],4))
P=np.c_[np.ones(n),rr,expit(p[:,-1])]; C=np.eye(4)[lab]; center=np.array([learn_cal([tr,te])[g][0] for g in lab]); invscale=np.array([1/learn_cal([tr,te])[g][1] for g in lab]);
models={'baseline':(np.zeros((n,0)),0.),'sigmoid':(P,.01),'sigmoid+panel':(np.c_[P,center,invscale],.1),'sigmoid+lab':(np.c_[P,C,C*(rr[:,None]-theta[2])],.2),'sigmoid+rawscore':(np.c_[P,rr*invscale,center*invscale],.1),'sigmoid+features':(np.c_[P,xnew],.3),'sigmoid+missing':(np.c_[P,~np.isfinite(raw)],.4),'sigmoid+lab-only':(np.c_[P,C],.2)}
for key,(F,reg) in models.items():
 def fit(mask):
  k,q,amp=profile(mask); co=np.linalg.lstsq(np.vstack([F[mask],np.sqrt(reg)*np.eye(F.shape[1])]),np.r_[(amp-rr)[mask],np.zeros(F.shape[1])],rcond=None)[0]; return target-(rr+F@co)[:,None]*q,co
 def cv(group):
  e=target.copy()
  for g in np.unique(group):
   testmask=group==g;e[testmask]=fit(~testmask)[0][testmask]
  return np.round([np.sqrt(np.mean(e**2)),max(np.sqrt(np.mean(e[lab==g]**2)) for g in np.unique(lab)),np.percentile(abs(e[:,:-1]),90),np.percentile(abs(e[:,-1]),90)],5)
 print(key,'LOLO / LOCO',cv(lab),cv(np.arange(n)),'correction',fit(np.ones(n,bool))[1].round(4))
