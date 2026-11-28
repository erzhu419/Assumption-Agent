import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); import upstream_transducer as u
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
 c=item['dataset']['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; k=u._fit_bridge(c); p=x[:,u._PARTNER]; F=u._design(x,p); e=y-(F*k).sum(-1); m=np.max(abs(y),axis=1)<5
 print('normalized coefficients',k[:,1:]/k[:,1,None]); print('largest residual rows', np.c_[np.arange(len(x)),x,e][m][np.argsort(np.max(abs(e[m]),axis=1))[-10:]])
 corr=np.stack((np.sin(2.5*(x+p))-np.sin(2.5*x)-np.sin(2.5*p),np.sin(2.5*(x-p))-np.sin(2.5*x)+np.sin(2.5*p),x*p),axis=-1)
 for scheme in ('interleaved','blocked','load radius'):
  ids=np.arange(len(x)); groups=ids%5 if scheme=='interleaved' else ids*5//len(x) if scheme=='blocked' else np.argsort(np.argsort(np.max(abs(x),axis=1)))*5//len(x)
  for extra in (False,True):
   D=np.concatenate((F,corr),axis=-1) if extra else F
   sw=c['single_channel_sweeps']; s=sw['true_stress']; raw=sw['cmd_stress']; labs=sw['lab_id']; A=u._design(s,np.zeros_like(s)); A=np.c_[A,np.zeros((len(s),3))] if extra else A
   pred=np.full_like(y,np.nan); loaderr=np.full_like(y,np.nan); weights=[]
   from scipy.optimize import least_squares
   for f in range(5):
    tr=m&(groups!=f); te=m&(groups==f); a=(abs(raw)<5)&(np.arange(len(s))%5!=f)
    b=np.array([np.linalg.lstsq(np.r_[D[tr,j],A[a&(labs==j)]],np.r_[y[tr,j],raw[a&(labs==j)]],rcond=None)[0] for j in range(4)]); weights.append(b)
    pred[te]=(D[te]*b).sum(-1)-y[te]
    for i in np.flatnonzero(te):
     def fun(h):
      p=h[u._PARTNER]; d=u._design(h,p)
      if extra: d=np.c_[d,np.sin(2.5*(h+p))-np.sin(2.5*h)-np.sin(2.5*p),np.sin(2.5*(h-p))-np.sin(2.5*h)+np.sin(2.5*p),h*p]
      return (d*b).sum(-1)-y[i]
     loaderr[i]=least_squares(fun,u._invert(y[i:i+1],b[:,:5])[0],gtol=1e-13).x-x[i]
   print(scheme, 'extra',extra,'inverse max/RMSE',np.nanmax(abs(loaderr)),np.sqrt(np.nanmean(loaderr**2)), 'forward max',np.nanmax(abs(pred)))
   if extra: print('correction coefficient min/max',np.min(weights,axis=0)[:,-3:],np.max(weights,axis=0)[:,-3:])
