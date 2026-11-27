import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_s00_solver.py'); globals().update({k:v for k,v in s.items() if k.startswith('_') and not k.startswith('__')})
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; b=_bridge(d['calibration_load_cells']); xt,xe=_features(tr,te); _,xg=_features({k:np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')},g); ta,th=_material(tr,b,xt,dt,t); p,h=_loads(g,b); cv,rt=_basis(ta,p,h,xg,dt,t); Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; lab=g['lab_id']; n=len(lab)
def T(q): return np.r_[np.c_[-np.expm1(-t[:,None]/q),t], [np.r_[1/q,1.]]]
def hist(q):
 z=np.zeros((n,len(q)))
 for w,a in zip(dt,p.T): z+=-np.expm1(-w/q)*(a[:,None]-z)
 return z
xd=_features(g,g)[0]; drift=th[2]+xd@th[3:]; F=T(ta)[:,:2]; R=Y-h[:,None]*(F@th[:2]); time=np.r_[t,0.]; print('time',time); print('empirical drift kernel',drift@R/(drift@drift)); print('fixed spectral coefficients',np.linalg.lstsq(T(ta),drift@R/(drift@drift),rcond=None)[0]); print('counts by lab',np.unique(lab,return_counts=True)); print('diag missing',np.sum(~np.isfinite(g['coupon_features_raw']),0))
def sat(q): return np.r_[-np.expm1(-t/q),1/q]
L=np.r_[t,1.]; ex=F.T
models={'empirical':(lambda a:np.eye(len(time)),[]),'branches':(lambda a:F,[]),'branches+linear':(lambda a:np.c_[F,L],[]),'single saturation':(lambda a:sat(np.exp(a[0]))[:,None],[np.log(1.5)]),'single+linear':(lambda a:np.c_[sat(np.exp(a[0])),L],[np.log(1.5)]),'fast+free saturation':(lambda a:np.c_[ex[0],sat(np.exp(a[0]))],[np.log(3)]),'fixed fast+slow+free':(lambda a:np.c_[F,sat(np.exp(a[0]))],[np.log(1.5)]),'fast+free+linear':(lambda a:np.c_[ex[0],sat(np.exp(a[0])),L],[np.log(3)])}
rm=lambda a:np.sqrt(np.mean(a*a));
for name,(basis,seed) in models.items():
 def fit(m):
  def proj(a):
   V=basis(a); y=drift[m]@R[m]/(drift[m]@drift[m]); c=np.linalg.lstsq(V,y,rcond=None)[0]; return V@c,c
  a=least_squares(lambda a:(R[m]-drift[m,None]*proj(a)[0]).ravel(),seed,bounds=(np.log(.055),np.log(24.9)),max_nfev=150).x if seed else []
  return proj(a),np.exp(a)
 E=np.zeros_like(Y)
 for l in np.unique(lab):
  j=lab==l; (k,c),_=fit(~j); E[j]=R[j]-drift[j,None]*k
 (k,c),a=fit(np.ones(n,bool)); print(name,'params',np.round(a,5),np.round(c,5),'OOF worst',np.round(np.max([[rm(E[lab==l,:-1]),rm(E[lab==l,-1])] for l in np.unique(lab)],0),6),'early/late',rm(E[:,:3]),rm(E[:,-4:-1]))
print('train/test-only drift difference',np.quantile(np.abs((_features(te,te)[0]-xe)@th[3:]),[0,.5,.9,1])); print('source bridge and features:')
