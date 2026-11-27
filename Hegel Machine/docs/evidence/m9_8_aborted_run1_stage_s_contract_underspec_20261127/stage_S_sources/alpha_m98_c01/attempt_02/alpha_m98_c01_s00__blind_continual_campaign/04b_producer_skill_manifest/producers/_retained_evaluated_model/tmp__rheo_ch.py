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
from itertools import combinations
pr=panel({k:np.concatenate([v[k] for v in (tr,ch,te)]) for k in ('lab_id','ref_coupons')}); xx,_=features(ch,pr,fill); E=y(ch)-mat(ch,xx,tau)@b; z=np.r_[t,1]; slope=E@z/(z@z); clean=E-slope[:,None]*z; P,H=loads(ch); g=ch['lab_id']; I=np.eye(4)[g]; print('row lab, correction slope, max departure from slope, standardized features, hold total:');print(np.c_[g,slope,np.max(abs(clean),axis=1),xx[:,1:],H.sum(1)].round(4));print('slope-removed medabs',np.median(abs(clean),axis=0).round(5));
# Compare transferable drift laws and panel conventions without using held-out responses.
for name,ref in [('train',trainref),('pooled',pr),('combined',panel({k:np.concatenate([v[k] for v in (tr,te)]) for k in ('lab_id','ref_coupons')}))]:
 xch,_=features(ch,ref,fill); xtr,_=features(tr,ref,fill); opt=least_squares(lambda v:(mat(tr,xtr,np.exp(v[:2]))@v[2:]-y(tr)).ravel(),theta,loss='cauchy',f_scale=.025); tt=np.exp(opt.x[:2]); bb=opt.x[2:]; E=y(ch)-mat(ch,xch,tt)@bb; slope=E@z/(z@z); print('PANEL',name,'baseline medabs',np.median(abs(E[:,:-1])).round(4),np.median(abs(E[:,-1])).round(4));
 cand={'none':xch[:,:0], 'intercept':xch[:,:1],'features':xch,'features+stress':np.c_[xch,H.sum(1)],'feature-quadratic':np.c_[xch,xch[:,1:]**2],'lab drift':np.c_[xch,I[:,1:]],'lab varying slope':np.c_[xch,(I[:,:,None]*xch[:,None,1:]).reshape(len(g),-1)]}
 for key,F in cand.items():
  def run(tridx,teidx):
   if not F.shape[1]: return np.zeros(len(teidx))
   a=F[tridx]; target=slope[tridx]; reg=.05 if key.startswith('lab') else .015
   init=np.linalg.lstsq(np.r_[a,reg*np.eye(a.shape[1])],np.r_[target,np.zeros(a.shape[1])],rcond=None)[0]; q=least_squares(lambda b:np.r_[a@b-target,reg*b],init,loss='soft_l1',f_scale=.03).x
   return F[teidx]@q
  err=[]
  for folds in [np.arange(len(g)),g,ch['stratum']]:
   e=E.copy();
   for label in np.unique(folds):
    a=np.flatnonzero(folds!=label); h=np.flatnonzero(folds==label); e[h]-=run(a,h)[:,None]*z
   err.append([np.mean(abs(e[:,:-1])),np.mean(abs(e[:,-1])),max(np.mean(abs(e[g==l,:-1])) for l in np.unique(g)),max(np.mean(abs(e[g==l,-1])) for l in np.unique(g))]);
  print(key,'LOO/Lab/Stratum [curve,rate,worst lab curve,rate]',np.array(err).round(4).tolist())
