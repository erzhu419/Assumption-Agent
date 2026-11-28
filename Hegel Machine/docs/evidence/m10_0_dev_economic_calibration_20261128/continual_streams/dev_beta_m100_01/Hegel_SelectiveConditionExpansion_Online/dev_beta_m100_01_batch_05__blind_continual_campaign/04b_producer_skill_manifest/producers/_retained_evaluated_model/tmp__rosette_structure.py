import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; axis=np.count_nonzero(X,axis=1)==1
clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<5
idx=np.flatnonzero(clean); np.random.default_rng(41).shuffle(idx); folds=np.array_split(idx,5)
sb=c['single_bank_sweeps']; banks=sb['bank_id']; u=sb['ref_signal']; v=sb['raw_readout'];
for j in range(4):
 m=(banks==j)&(abs(v)<5); t=u[m]; y=v[m]; p=least_squares(lambda p:p[0]+p[1]*np.tanh(p[2]*t)-y,[0,1.5,1.]).x;print('single bank',j,p,np.round(p[0]+p[1]*np.tanh(p[2]*t)-y,5))
for kind,fun in [('tanh',np.tanh),('atan',np.arctan),('sqrt',lambda z:z/np.sqrt(1+z*z)),('erf',erf)]:
 for variant in ['separate','linear','shift','shiftlinear','quadratic','pair','paired_shiftlinear']:
  def fit(i):
   pred=[]; params=[]
   for j in range(4):
    q=[j,j^2]; x=X[i][:,q]; y=Y[i,j]
    def db(xx,p):
     z=xx*np.exp(p[:2]);
     if 'shift' in variant: z=z+p[2:4]
     d=np.c_[np.ones(len(z)),fun(z)]
     if 'linear' in variant: d=np.c_[d,xx]
     if variant=='quadratic': d=np.c_[d,xx**2]
     if variant in ('pair','paired_shiftlinear'): d=np.c_[d,xx[:,0]*xx[:,1]]
     return d
    def coef(p):return np.linalg.lstsq(db(x,p),y,rcond=None)[0]
    lim=([-.0-4]*2+([-2]*2 if 'shift' in variant else []),[2]*2+([2]*2 if 'shift' in variant else [])); p=least_squares(lambda p:db(x,p)@coef(p)-y,np.zeros(len(lim[0])),bounds=lim,max_nfev=100,gtol=1e-10).x
    b=coef(p); pred.append((q,p,b));params.append(np.r_[p,b])
   return lambda xx: np.column_stack([db(xx[:,q],p)@b for q,p,b in pred]), params
  pr,p=fit(clean); e=pr(X)-Y; cv=np.zeros_like(Y)
  for test in folds:
   tr=clean.copy();tr[test]=False;cv[test]=fit(tr)[0](X[test])-Y[test]
  print(kind,variant,'train',np.round(np.sqrt(np.mean(e[clean]**2,0)),7),'axis/mixed',np.round([np.sqrt(np.mean(cv[clean&m]**2)) for m in (axis,~axis)],7));
  if kind=='tanh': print('ch0 params',np.round(p[0],5))
