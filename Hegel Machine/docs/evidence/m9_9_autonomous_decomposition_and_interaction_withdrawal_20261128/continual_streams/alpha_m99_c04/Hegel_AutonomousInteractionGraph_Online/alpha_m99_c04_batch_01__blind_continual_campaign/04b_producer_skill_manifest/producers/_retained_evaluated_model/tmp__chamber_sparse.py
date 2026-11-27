import numpy as np,pickle,runpy
from scipy.optimize import least_squares
np.set_printoptions(precision=6,suppress=True)
a=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c04_s00_solver.py'); i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; s=a['_stress_map'](ds['calibration_load_cells']); xt,xe=a['_features'](tr,te)
def empty(g): return {k:v[:0] for k,v in g.items()}
for name,xx in [('joint',xt),('train only',a['_features'](tr,empty(tr))[0])]:
 q,b=a['_fit'](tr,xx,s,dt,t); A,R=a['_design'](tr,xx,s,dt,t,q); e=A@b-(tr['strain_curves_obs']-tr['strain_0'][:,None]); er=R@b-tr['instant_rate_obs']; print('intrinsic',name,q,b,'median',np.median(abs(e)),np.median(abs(er)))
q,b=a['_fit'](tr,xt,s,dt,t); x=a['_features'](d,empty(d))[0]; r=b[2]+x@b[3:]; p=s[0]+d['past_channel_raw']@s[1:]; h=s[0]+d['hold_channel_raw']@s[1:]; y=np.c_[d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs']]; O=np.ones(len(h)); age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; z=p@(-np.expm1(-dt[:,None]/q)*np.exp(-age[:,None]/q)); w=np.r_[np.ones(len(t)),2.5]; n,k=y.shape; ids=np.arange(n); labs=d['lab_id']; miss=np.isnan(d['coupon_features_raw']).any(1)
models={
 'two affine amplitudes': [np.c_[O,h,r],np.c_[O,h,r],None],
 'two hold and drift': [np.c_[h,r],np.c_[h,r],None],
 'hold drift plus small drift': [np.c_[h,r],np.c_[h,r],np.c_[O,r]],
 'coupled memory': [np.c_[h,r,z],np.c_[h,r,z],None],
 'two affine + slow memory': [np.c_[O,h,r],np.c_[O,h,r,z],None],
 'two affine + drift': [np.c_[O,h,r],np.c_[O,h,r],np.c_[O,r]],
 'two affine + separate raw features': [np.c_[O,h,x],np.c_[O,h,x],None],
 'two affine + held load interaction': [np.c_[O,h,r,h*r],np.c_[O,h,r,h*r],None],
}
def fit(F,j,mode):
 def des(p):
  B=np.vstack([np.c_[-np.expm1(-t[:,None]/p),t],np.r_[1/p,1]])
  return np.concatenate([D[:,None,:]*B[None,:,l,None] for l,D in enumerate(F) if D is not None],axis=-1)
 def res(logp,ret=False):
  A=des(np.exp(logp)); W=(A[j]*w[None,:,None]).reshape(len(j)*k,-1); Y=(y[j]*w).ravel(); c=np.linalg.lstsq(W,Y,rcond=None)[0]; return (A@c,c) if ret else W@c-Y
 lp=np.log(q)
 if mode=='free': lp=least_squares(res,lp,bounds=(np.log([.4,1.5]),np.log([.8,8.])),diff_step=1e-4).x
 if mode=='slow': lp[1]=least_squares(lambda v:res([lp[0],v[0]]),lp[1:],bounds=(np.log(1.5),np.log(8.))).x[0]
 return (*res(lp,True),np.exp(lp))
def metric(e): return np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))
for name,F in models.items():
 for mode in ['fixed','slow','free']:
  yh,c,qp=fit(F,ids,mode); cv=np.empty_like(y)
  for l in np.unique(labs):
   m=labs==l; cv[m]=fit(F,ids[~m],mode)[0][m]-y[m]
  print(name,mode,'poles',qp,'fit',np.round(metric(yh-y),5),'oof',np.round(metric(cv),5),'worst lab',np.round(np.max([metric(cv[labs==l]) for l in np.unique(labs)],axis=0),5),'missing',np.round(metric(cv[miss]),5),'coef',c)
for name,g in [('train',tr),('test',te),('diagnostic',d)]:
 xx=a['_features'](g,empty(g))[0]; print(name,'local feature quantiles',np.round(np.quantile(xx,[.1,.5,.9],axis=0),3))
