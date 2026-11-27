import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_06_solver.py'
s=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=b['dataset']; g=b['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; cal=d['calibration_load_cells']; x=cal['true_channel_loads']; y=cal['bridge_raw']
for model in ('affine','compact'):
 es=[]
 for fold in range(5):
  test=np.arange(len(x))%5==fold; keep=~test
  for _ in range(5):
   p=[]
   for j in range(4):
    a=m._bridge_terms(x)
    if model=='compact': a=np.column_stack((a,x[:,j]**3,x[:,j]*x[:,(j+1)%4]))
    p.append(a@np.linalg.lstsq(a[keep],y[keep,j],rcond=None)[0])
   e=np.linalg.norm(np.array(p).T-y,axis=1); keep=(~test)&(e<5*np.median(e[keep]))
  es.append(e[test])
 print('bridge',model,'worst fold median',max(map(np.median,es)),'pooled q',np.quantile(np.concatenate(es),[.5,.9,.95,1]))
c=m._fit_bridge(cal); print('coeff',c.round(6)); print('bridge errors',np.quantile(np.linalg.norm(m._bridge_forward(x,c)-y,axis=1),[.5,.9,.95,1]))
X=m._aligned_features(tr,te); print('X',X.shape)
train=m._coupon(tr,X[:len(tr['lab_id'])],c); taus,theta=m._fit(dt,t,train,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);print('fit',taus,theta)
def combine(a,b): return {k:np.concatenate([a[k],b[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
def frozen(g):
 p=combine(tr,te); x=g['coupon_features_raw'].copy()
 for lab in np.unique(g['lab_id']):
  r=p['ref_coupons'][p['lab_id']==lab]; i=g['lab_id']==lab
  x[i]=(x[i]-np.median(np.mean(r,axis=0)))/np.median(np.std(r,axis=0))
 _,j,n=np.unique(np.column_stack((p['lab_id'],p['stratum'])),axis=0,return_inverse=True,return_counts=True); w=1/n[j,None]
 raw=p['coupon_features_raw']; fill=np.sum(X[:,1:]*w*~np.isnan(raw),axis=0)/np.sum(w*~np.isnan(raw),axis=0)
 return np.column_stack((np.ones(len(x)),np.where(np.isnan(x),fill,x)))
cy=g['strain_curves_obs']-g['strain_0'][:,None]; ry=g['instant_rate_obs']; G=-np.expm1(-t[:,None]/taus)
for label,z in [('frozen',frozen(g)),('joint_td',m._aligned_features(tr,g)[-len(cy):]),('joint_ttd',m._aligned_features(combine(tr,te),g)[-len(cy):])]:
 group=m._coupon(g,z,c); past,h,_=group; A,R=m._design(taus,dt,t,group); bas=A@theta; br=R@theta; dr=z@theta[2:]; reset,rr=m._chamber(taus,theta,t,h,z)
 def score(name,p,q):
  print(label,name,'curve median/q90/RMSE',np.median(abs(p-cy)),np.quantile(abs(p-cy),.9),np.sqrt(np.mean((p-cy)**2)),'rate median/RMSE',np.median(abs(q-ry)),np.sqrt(np.mean((q-ry)**2)))
 score('base',bas,br);score('reset',reset,rr)
 score('preserve',bas+dr[:,None]*(G@(taus*np.array([.62686,.60591]))-t*(.62686+.60591)),br)
 # Profile separate initial memory retention and drift redistribution in both branches.
 reset0=(h[:,None]*theta[:2])@G.T+dr[:,None]*t; rr0=h@(np.ones(1)) if False else h*np.sum(theta[:2]/taus)+dr
 mem=bas-reset0; Z=np.stack([(A[:,:,j]-(h[:,None]*G[None,:,j]))*theta[j] for j in range(2)]+[dr[:,None]*(G[:,j]*taus[j]-t) for j in range(2)],axis=-1)
 W=np.column_stack([(R[:,j]-h/taus[j])*theta[j] for j in range(2)]+[np.zeros(len(h))]*2)
 def fit(rows):
  def f(q):return np.r_[((reset0[rows]+Z[rows]@q-cy[rows])/.001).ravel(),(rr0[rows]+W[rows]@q-ry[rows])/.002]
  return least_squares(f,[.5,.5,.5,.5],loss='cauchy').x
 q=fit(np.arange(len(cy))); score('profile '+str(q.round(4)),reset0+Z@q,rr0+W@q)
 print('LOLO chamber q',[(lab,fit(g['lab_id']!=lab).round(4)) for lab in np.unique(g['lab_id'])])
 print('offset medians by time',np.median(reset0+Z@q-cy,axis=0).round(6),'rate', (rr0+W@q-ry).round(5))
