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
from itertools import combinations
aux=te['aux_chamber_telemetry']; print('aux',type(aux),np.shape(aux)); print(str(aux)[:850]); print('feature and load distributions',[(np.round(np.nanmean(ob['coupon_features_raw'],0),3),np.round(np.nanstd(ob['coupon_features_raw'],0),3)) for ob in [tr,te,dg]])
tt=np.r_[t,0.]; response=Z(1.0); drift=theta[2]+xnew@theta[3:]; amp=target@response/(response@response); err=amp-drift; lab=dg['lab_id']; st=dg['stratum']; complete=np.isfinite(dg['coupon_features_raw']).all(1); print('fixed-history shape RMS',np.sqrt(np.mean((target-amp[:,None]*response)**2))); print('amplitude residual by coupon',np.round(np.c_[lab,st,complete,err,p],3))
F=np.c_[h,p,z,np.diff(p,axis=1),p.mean(1),p.std(1),np.abs(p).mean(1),drift,xnew,st]; names=['hold']+[f'p{i}' for i in range(p.shape[1])]+['z0','z1']+[f'dp{i}' for i in range(p.shape[1]-1)]+['meanpast','stdpast','abspast','drift','x0','x1','x2','x3','stratum']; print('amplitude correlations complete',sorted(zip(names,np.round(np.corrcoef(np.c_[F[complete],err[complete]].T)[:-1,-1],3)),key=lambda z:-abs(z[1])))
# Select small corrections in each training fold, then predict its excluded lab.
for xx in [xnew,xd]:
 drift=theta[2]+xx@theta[3:]; err=amp-drift
 pools={'none':[], 'history':[names.index(nm) for nm in ['hold','p0','p1','p2','p3','p4','p5','z0','z1','meanpast','stdpast','abspast']], 'features':list(range(len(names)-6,len(names)-1)), 'mixed':list(range(len(names)))}
 for name, inds in pools.items():
  for k in ([0] if name=='none' else [1,2,3]):
   configs=[]
   for sel in combinations(inds,k):
    X=np.c_[np.ones(n),F[:,sel]]; configs.append((sel,X))
   cv=np.zeros(n); picks=[]
   for g in np.unique(lab):
    mask=lab!=g; choices=[]
    for sel,X in configs:
     c=np.linalg.lstsq(X[mask],err[mask],rcond=None)[0]; choices.append((np.mean((X[mask]@c-err[mask])**2),sel,X,c))
    _,sel,X,c=min(choices,key=lambda a:a[0]); cv[~mask]=(X@c-err)[~mask]; picks.append([names[j] for j in sel])
   print('amp CV',('joint' if xx is xnew else 'all'),name,k,'rms/worst lab',round(np.sqrt(np.mean(cv**2)),4),round(max(np.sqrt(np.mean(cv[lab==g]**2)) for g in np.unique(lab)),4),picks)
