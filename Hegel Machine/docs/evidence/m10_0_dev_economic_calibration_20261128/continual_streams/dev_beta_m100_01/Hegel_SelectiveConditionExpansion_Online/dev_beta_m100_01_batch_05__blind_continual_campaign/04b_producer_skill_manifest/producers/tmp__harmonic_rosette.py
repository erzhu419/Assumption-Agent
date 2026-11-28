import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; pair=np.arange(4)^2; axis=np.count_nonzero(X,axis=1)==1; clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<5
sb=c['single_bank_sweeps']; U,V,B=[sb[k] for k in ('ref_signal','raw_readout','bank_id')]; keep=abs(V-np.median(V))<5

def fit(rows,banks,extra):
 def design(x,z,f):
  parts=[np.ones_like(x),x,np.sin(f[0]*x),z,np.cos(f[1]*z)-1]
  if extra: parts.extend((np.sin(f[1]*z),np.cos(f[0]*x)-1))
  return np.stack(parts,-1)
 def profile(f,ret=False):
  pars=[]; err=[]
  for j,p in enumerate(pair):
   use=banks&(B==j); a=np.r_[X[rows,j],U[use]]; b=np.r_[X[rows,p],np.zeros(use.sum())]; v=np.r_[Y[rows,j],V[use]]; d=design(a,b,f); le=np.linalg.lstsq(d,v,rcond=None)[0];pars.append(le);err.extend(d@le-v)
  return np.asarray(pars) if ret else np.asarray(err)
 f=least_squares(profile,[2.5,2.0],bounds=(.7,4),gtol=1e-13,ftol=1e-12).x; pars=profile(f,True); return f,pars,lambda x:np.einsum('nck,ck->nc',design(x,x[:,pair],f),pars)
idx=np.flatnonzero(clean); np.random.default_rng(41).shuffle(idx)
for extra in (False,True):
 f,p,fun=fit(clean,keep,extra); print('extra',extra,'frequencies',f,'coef',p,'train RMSE',np.sqrt(np.mean((fun(X[clean])-Y[clean])**2))); cv=np.zeros_like(Y)
 for tst in np.array_split(idx,5):
  tr=clean.copy();tr[tst]=False; cv[tst]=fit(tr,keep,extra)[2](X[tst])-Y[tst]
 print('cv axis/mixed',[(np.sqrt(np.mean(cv[m&clean]**2)),np.max(abs(cv[m&clean]))) for m in (axis,~axis)]); print('group holdout axis-only error on mixed, mixed-only error on axes',[np.sqrt(np.mean((fit(clean&m,keep,extra)[2](X[clean&~m])-Y[clean&~m])**2)) for m in (axis,~axis)])
 print('single bank CV',max(np.sqrt(np.mean([(fit(clean,keep&(B!=j),extra)[2](np.eye(4)[j]*U[q])[:,j]-V[q]).dot(fit(clean,keep&(B!=j),extra)[2](np.eye(4)[j]*U[q])[:,j]-V[q])/len(q)])) for j in range(4) for q in [np.flatnonzero(keep&(B==j))]) if False else '')
