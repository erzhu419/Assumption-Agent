import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c02_batch_06_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te=d['train'],d['test']; c=p['campaign_diagnostics']
q=np.load('/tmp/chamber_inputs.npz'); globals().update({k:q[k] for k in q.files}); obs=np.c_[Y,V]; K=np.vstack([kernel,1/T]); tail=np.cumsum(dt[::-1])[::-1]-dt
print('counts/refs',[(len(x['lab_id']),x['ref_coupons'].shape) for x in (tr,te,c)], 'times',t)

def prep(panels):
 refs=np.concatenate([u['ref_coupons'] for u in panels]); labs=np.concatenate([u['lab_id'] for u in panels]); groups=np.concatenate([u['stratum'] for u in panels]); x=np.concatenate([u['coupon_features_raw'] for u in panels]).copy(); zc=c['coupon_features_raw'].copy()
 for l in np.unique(lab):
  sel=labs==l; R=refs[sel]; a=np.nanmedian(np.nanmean(R,axis=0)); b=np.nanmedian(np.nanstd(R,axis=0)); x[sel]=(x[sel]-a)/b; zc[lab==l]=(zc[lab==l]-a)/b
 _,inv,cnt=np.unique(np.c_[labs,groups],axis=0,return_inverse=True,return_counts=True); w=(1/cnt[inv])[:,None]; imp=np.nansum(w*x,axis=0)/np.sum(w*np.isfinite(x),axis=0); return np.where(np.isfinite(zc),zc,imp)

def analyze(name,fx):
 delta=H[2]+fx@H[3:]
 for mode in ('plain','gate','hold','relax'):
  def pred(p):
   if mode=='plain': wait,a,b,g,s,fac=0,0,p[0],p[1],1,np.ones(2)
   elif mode=='gate': wait,a,b,g=p; s=1; fac=np.ones(2)
   elif mode=='hold': wait,a,b,g,s=p; fac=np.ones(2)
   else: wait,a,b,g,s,*fac=p
   A=(s*hold[:,None]-(past@W)*np.exp(-wait/T)*fac)*H[:2]; return A@K.T+delta[:,None]*np.r_[a*t-b*g*np.expm1(-t/g),a+b]
  def fit(mask):
   x,lo,hi=([1.,1.],[.2,.2],[2.,4.]) if mode=='plain' else ([0.,0.,1.,1.],[-1,-.4,.2,.2],[3,.6,2,4]); extras=0 if mode in ('plain','gate') else (1 if mode=='hold' else 3); x=x+[1.]*extras; lo=lo+[.75]*extras; hi=hi+[1.35]*extras; return least_squares(lambda p:(pred(p)[mask]-obs[mask]).ravel(),x,bounds=(lo,hi),loss='soft_l1',f_scale=.009).x
  f=fit(np.ones(len(Y),bool)); r=pred(f)-obs; oof=np.empty_like(obs)
  for l in np.unique(lab):
   j=lab==l; oof[j]=pred(fit(~j))[j]-obs[j]
  er=np.array([[np.sqrt(np.mean(oof[lab==l,:-1]**2)),np.sqrt(np.mean(oof[lab==l,-1]**2))] for l in np.unique(lab)]); print(name,mode,np.round(f,4),'fit',np.round([np.sqrt(np.mean(r[:,:-1]**2)),np.sqrt(np.mean(r[:,-1]**2))],4),'worst',er.max(0).round(4),'lab residual means',np.round([r[lab==l,-1].mean() for l in np.unique(lab)],4))
for name,panels in [('trte',[tr,te]),('trc',[tr,c]),('c',[c]),('all',[tr,te,c]),('tr',[tr]),('te',[te])]: analyze(name,prep(panels))
