import pickle, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=6,suppress=True,linewidth=170)
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; b=c['single_bank_sweeps']; a=b['ref_signal']; by=b['raw_readout']; ix=b['bank_id']
print('ref range',x.min(0),x.max(0)); obs=s['y_obs_raw'].reshape(-1,4); print('obs range',obs.min(0),obs.max(0)); print('bank layout',s['row_bank_id'].reshape(-1,4)[:6]); print('Z range',s['Z_nuisance'].min(0),s['Z_nuisance'].max(0))
co=[]
for k in range(4):
 use=ix==k; iso=np.zeros((use.sum(),4));iso[:,k]=a[use]; yy=np.r_[y[:,k],by[use]]; xx=np.vstack((x,iso)); print('BANK',k)
 for typ in ('own','all','full'):
  z=np.column_stack((np.ones(len(xx)),xx,xx[:,k]**3 if typ=='own' else xx**3))
  names=['1',*map(str,range(4)),f'{k}^3'] if typ=='own' else ['1',*map(str,range(4)),*[f'{j}^3' for j in range(4)]]
  if typ=='full':
   for i in range(4):
    for j in range(i,4):
     z=np.column_stack((z,xx[:,i]*xx[:,j])); names.append(f'{i}{j}')
     for l in range(j,4):
      if i==l: continue
      z=np.column_stack((z,xx[:,i]*xx[:,j]*xx[:,l])); names.append(f'{i}{j}{l}')
  def fit(m):
   t=least_squares(lambda t:(z@t-yy)[m],np.linalg.lstsq(z[m],yy[m],rcond=None)[0],jac=lambda t:z[m],loss='cauchy',f_scale=.018).x
   clean=m&(abs(yy-z@t)<.001)
   return np.linalg.lstsq(z[clean],yy[clean],rcond=None)[0]
  t=fit(np.ones(len(yy),bool)); e=yy-z@t; cv=np.empty_like(e)
  for j in range(5):
   m=np.arange(len(e))%5!=j; cv[~m]=yy[~m]-z[~m]@fit(m)
  print(typ,[(n,round(v,7)) for n,v in zip(names,t) if abs(v)>.0001], 'CV inlier max',max(abs(cv[abs(e)<.001])), 'outliers',np.flatnonzero(abs(e)>.001))
  if typ=='full':co.append((names,t))
