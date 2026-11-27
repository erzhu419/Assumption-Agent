from probe import *
from scipy.optimize import least_squares
B,I=curve[:,:,:2]@theta[:2],rate[:,:2]@theta[:2]
E=Y-B
labs=cd['lab_id']; F=np.stack([tr[2]@theta[2:]],1)
# Profile coupon amplitude to separate temporal shape from feature calibration.
kernels={
 'exp':(lambda a: -np.expm1(-a*t)/a,[.9]),
 'log':(lambda a: np.log1p(a*t)/a,[1.8]),
 'rational':(lambda a: t/(1+a*t),[.6]),
 'exp+linear':(lambda a,b: (1-b)*(-np.expm1(-a*t)/a)+b*t,[.9,0]),
 'rational-power':(lambda a,b: t/(1+a*t)**b,[.6,1]),
 'exp-quadratic':(lambda a,b: t*np.exp(-a*t/(1+b*t)),[.5,.25]),
 'exp-taper':(lambda a,b: -np.expm1(-a*t)/a*np.exp(-b*t),[.9,0]),
}
for name,(fn,start) in kernels.items():
 def calc(v,e):
  f=fn(*v); q=e@f/(f@f); return q,f
 def fit(e):
  def obj(v):
   q,f=calc(v,e); return (e-q[:,None]*f).ravel()
  return least_squares(obj,start,loss='soft_l1',f_scale=.004,bounds=([.0001]+([-0.2] if len(start)>1 else []),[15]+([5.] if len(start)>1 else []))).x
 v=fit(E); q,f=calc(v,E); sl=[]
 for l in np.unique(labs):
  m=labs==l; v0=fit(E[~m]); q0,f0=calc(v0,E[m]); sl.append([np.sqrt(np.mean((E[m]-q0[:,None]*f0)**2)),np.sqrt(np.mean((R[m]-I[m]-q0)**2))])
 print(name,'params',v.round(5),'shape RMSE',np.sqrt(np.mean((E-q[:,None]*f)**2)),'worst OOF shape/rate',np.max(sl,axis=0).round(5),'amp minus drift RMSE',np.sqrt(np.mean((q-drift)**2)))
 if name=='exp':
  print('amp comparisons lab, missing, base, estimated, observed-rate',np.c_[labs,np.isnan(cd['coupon_features_raw']).sum(1),drift,q,R-I].round(4)); print('f',f.round(5)); print('empirical f',((q@E)/(q@q)).round(5))
for splits in [(c['train'],c['test']), (c['train'],cd), (c['train'],c['test'],cd)]:
 x0=prior._features(*splits); tc=prior._coupon(c['train'],x0[:n],br); a,b=prior._fit(tc,dt,t,y,r)
 print('feature configuration',list(map(lambda s:len(s['lab_id']),splits)),a.round(5),b.round(5))
 for l in np.unique(labs):
  print('panel',l,[(np.median(s['ref_coupons'][s['lab_id']==l].mean(0)).round(4),np.median(s['ref_coupons'][s['lab_id']==l].std(0)).round(4)) for s in splits])
