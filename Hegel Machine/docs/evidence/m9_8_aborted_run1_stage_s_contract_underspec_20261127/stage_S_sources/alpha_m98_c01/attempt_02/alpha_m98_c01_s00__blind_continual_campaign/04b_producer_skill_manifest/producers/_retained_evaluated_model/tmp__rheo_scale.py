import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr=D['train']; ch=D['chamber_verification_trials']; te=D['test']; dt=D['past_dt']; t=D['eval_times']; end=np.r_[np.cumsum(dt[::-1])[::-1],0]; C=D['calibration_load_cells']; A=C['bridge_raw']; B=C['true_channel_loads']; q=np.linalg.lstsq(np.c_[A,np.ones(len(A))],B,rcond=None)[0]
def panel(d):
 g=d['lab_id']; return {l:(np.median(d['ref_coupons'][g==l].mean(0)),np.median(d['ref_coupons'][g==l].std(0))) for l in np.unique(g)}
def raw(d,ref):
 x=d['coupon_features_raw'].copy();g=d['lab_id'];
 for l in np.unique(g):
  m,s=ref[l];x[g==l]=(x[g==l]-m)/s
 return x
def features(d,ref,fill=None):
 x=raw(d,ref)
 if fill is None:
  _,idx,n=np.unique(np.c_[d['lab_id'],d['stratum']],axis=0,return_counts=True,return_inverse=True); w=1/n[idx,None];fill=np.nansum(w*x,0)/np.sum(w*np.isfinite(x),0)
 return np.c_[np.ones(len(x)),np.where(np.isfinite(x),x,fill)],fill
def loads(d):return d['past_channel_raw']@q[:4]+q[4],d['hold_channel_raw']@q[:4]+q[4]
def mat(d,x,a):
 p,h=loads(d); r=h.sum(1)[:,None]-p.sum(2)@np.diff(np.exp(-end[:,None]/a),axis=0); return np.concatenate([r[:,None,:]*np.vstack([-np.expm1(-t[:,None]/a),1/a]),x[:,None,:]*np.r_[t,1][None,:,None]],axis=2)
def y(d):return np.c_[d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs']]
trainref=panel(tr); x,fill=features(tr,trainref); theta=np.r_[np.log([.55,4]),[.8,.5],np.zeros(5)]; opt=least_squares(lambda v:(mat(tr,x,np.exp(v[:2]))@v[2:]-y(tr)).ravel(),theta,loss='cauchy',f_scale=.025); tau=np.exp(opt.x[:2]); b=opt.x[2:]; print('tau',tau,'b',b,'fill',fill)
z=np.r_[t,1]; g=ch['lab_id']; u=np.unique(g); I=(g[:,None]==u); idx=np.arange(len(g));
for panel_name,ref in [('train',trainref),('pool',panel({k:np.concatenate([v[k] for v in (tr,ch,te)]) for k in ('lab_id','ref_coupons')}))]:
 xt,f=features(tr,ref); v=least_squares(lambda v:(mat(tr,xt,np.exp(v[:2]))@v[2:]-y(tr)).ravel(),theta,loss='cauchy',f_scale=.025).x; tt=np.exp(v[:2]); bb=v[2:]; X,_=features(ch,ref,f); E=y(ch)-mat(ch,X,tt)@bb; ds=E@z/(z@z); d0=X[:,1:]@bb[3:];
 print(panel_name,'baseline slope and slope correction:',np.c_[g,d0,ds].round(4).tolist());
 for name,F,pen in [('scale only',I*d0[:,None],.08),('scale with offset',np.c_[I,I*d0[:,None]],.08),('common offset with lab scale',np.c_[np.ones(len(g)),I*d0[:,None]],.08),('scale + individual slopes',np.c_[I,I*d0[:,None],(I[:,:,None]*X[:,None,1:]).reshape(len(g),-1)],np.r_[np.full(8,.08),np.full(16,.4)])]:
  p=np.broadcast_to(pen,(F.shape[1],)); print(name, 'coeffs',np.linalg.lstsq(np.r_[F,np.diag(p)],np.r_[ds,np.zeros(len(p))],rcond=None)[0][:8].round(3));
  for groups in (idx,ch['stratum'],g):
   er=E.copy()
   for label in np.unique(groups):
    j=groups==label; co=np.linalg.lstsq(np.r_[F[~j],np.diag(p)],np.r_[ds[~j],np.zeros(len(p))],rcond=None)[0]; er[j]-=(F[j]@co)[:,None]*z
   print(np.round([np.mean(abs(er[:,:-1])),np.mean(abs(er[:,-1])),max(np.mean(abs(er[g==l,:-1])) for l in u),max(np.mean(abs(er[g==l,-1])) for l in u)],4).tolist(),end=' ')
  print()
