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
pool={k:np.concatenate([v[k] for v in (tr,ch,te)]) for k in ('lab_id','ref_coupons')};pr=panel(pool); print('panels',trainref,pr,panel(te),panel(ch));
for ref in (trainref,pr):
 xx,_=features(ch,ref,fill); G=mat(ch,xx,tau); E=y(ch)-G@b; P,H=loads(ch); print('lab stratum, hold, raw X, residuals at t0.25, t1.27, t2.64, t4, initial rate'); print(np.c_[ch['lab_id'],ch['stratum'],H.sum(1),ch['coupon_features_raw'],E[:,[0,3,7,11,12]]].round(3)); print('CORR residual at t4 by hold ch, hold sum, features, stratum',np.corrcoef(np.c_[H,H.sum(1),xx[:,1:],ch['stratum'],E[:,11]].T)[-1,:-1].round(2));
 for name,d in [('tr',tr),('ch',ch)]:
  xx,_=features(d,ref,fill); e=y(d)-mat(d,xx,tau)@b; print(name,'median abs',np.median(abs(e),0).round(4),'median',np.median(e,0).round(4))
print('archive exists?');
