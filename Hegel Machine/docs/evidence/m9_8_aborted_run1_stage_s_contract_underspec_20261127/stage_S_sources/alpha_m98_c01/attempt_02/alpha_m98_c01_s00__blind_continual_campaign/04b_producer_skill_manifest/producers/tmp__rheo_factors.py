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
print('lab slopes and raw feature scales');
for l in np.unique(g):
 m=g==l; a=xx[m,1:]; raw_x=ch['coupon_features_raw'][m]; yy=slope[m]; print(l,'n',sum(m),'raw spread',np.nanstd(raw_x,axis=0).round(3),'correlations',np.corrcoef(np.c_[a,yy].T)[-1,:-1].round(3),'intercept and slopes',np.linalg.lstsq(np.c_[np.ones(sum(m)),a],yy,rcond=None)[0].round(4));
print('train/test aux correlations with X, stress, refs');
for d in [tr,te]:
 F,_=features(d,pr,fill); pp,hh=loads(d); u=d['aux_chamber_telemetry']; print(np.corrcoef(np.c_[u,F[:,1:],hh.sum(1),d['ref_coupons'].mean(1)].T)[:4,4:].round(2)); print('lab stats:',[(l,u[d['lab_id']==l].mean(0).round(2).tolist(),u[d['lab_id']==l].std(0).round(2).tolist()) for l in np.unique(d['lab_id'])]);
print('reference singular values by lab and panel');
for d in [tr,ch,te]:
 for l in np.unique(d['lab_id']):
  r=d['ref_coupons'][d['lab_id']==l]; print(l,np.mean(r,0).round(2),np.std(r,0).round(2));
print('chamber mixed slopes using trained baseline and pooled panels, leave one coupon out');
xch,_=features(ch,pr,fill); xtr,f=features(tr,pr); G=mat(ch,xch,tau); E=y(ch)-G@b; slope=E@z/(z@z)
for basis in [(),(0,),(2,),(0,2),(0,1,2,3)]:
 F=np.c_[xch,(I[:,:,None]*xch[:,None,np.array(basis,dtype=int)+1]).reshape(len(g),-1)]
 for pen in [.05,.25]:
  a=np.r_[np.full(5,.25),np.full(F.shape[1]-5,pen)]; oo=[]
  for j in range(len(F)):
   m=np.arange(len(F))!=j; coef=np.linalg.lstsq(np.r_[F[m],np.diag(a)],np.r_[slope[m],np.zeros(len(a))],rcond=None)[0]; oo.append(F[j]@coef)
  e=E-np.array(oo)[:,None]*z; print(basis,pen,np.mean(abs(e[:,:-1])).round(4),np.mean(abs(e[:,-1])).round(4),[round(np.mean(abs(e[g==l,:-1])),4) for l in np.unique(g)]);
