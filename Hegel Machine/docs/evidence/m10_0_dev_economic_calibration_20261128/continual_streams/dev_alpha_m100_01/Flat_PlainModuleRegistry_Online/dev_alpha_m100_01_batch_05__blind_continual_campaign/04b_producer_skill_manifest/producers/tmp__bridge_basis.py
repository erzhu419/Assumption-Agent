import pickle,json,numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; clean=np.max(abs(y),axis=1)<5; axis=np.count_nonzero(x,axis=1)==1
np.set_printoptions(precision=5,suppress=True)
for j in range(4):
 u,v=x[:,j],x[:,j^2]; z=y[:,j]
 print('\nCH',j)
 for n,m,mode in [(5,5,'pair'),(8,8,'pair'),(11,11,'pair'),(8,8,'all'),(8,8,'interact'),(8,8,'hinges'),(8,8,'split'),(8,8,'warp')]:
  U=chebvander(u/1.75,n); V=chebvander(v/1.75,m); A=np.c_[U,V[:,1:]]
  if mode=='all': A=np.c_[A,*[chebvander(x[:,k]/1.75,3)[:,1:] for k in range(4) if k not in (j,j^2)]]
  if mode=='interact': A=np.c_[A,*[U[:,k]*V[:,l] for k in range(1,4) for l in range(1,4) if k+l<=4]]
  if mode=='hinges': A=np.c_[U,np.maximum(v,0),np.minimum(v,0),np.maximum(v,0)**2,np.minimum(v,0)**2,np.maximum(v,0)**3,np.minimum(v,0)**3]
  err=np.zeros(len(x))
  def fit(tr,te):
   if mode not in ('warp','split'): return A[te]@np.linalg.lstsq(A[tr],z[tr],rcond=None)[0]
   def pred(p):
    b,a,k,c,d,h,l=p
    if mode=='split': return b+a*np.tanh(k*u)+c*np.tanh(d*np.minimum(v,0))+h*np.tanh(l*np.maximum(v,0))
    return b+a*np.tanh(k*u+c*u**3)+d*v+h*abs(v)+l*v**3
   p=least_squares(lambda p:(pred(p)-z)[tr],[0,1.8,1.1,.3,1,.2,1],max_nfev=300).x
   return pred(p)[te]
  for fold in range(5):
   te=clean&(np.arange(len(x))%5==fold); err[te]=fit(clean&~te,te)-z[te]
  print(n,m,mode,'rmse',*[round(np.mean(err[s]**2)**.5,6) for s in (clean,clean&axis,clean&~axis)],'max',round(max(abs(err)),6),'in',round(np.mean((fit(clean,clean)-z[clean])**2)**.5,6))
  if mode=='pair' and n==8:
   ix=np.argsort(abs(err))[-6:];print('worst i,x,e',np.c_[ix,x[ix],err[ix]])
print('Telemetry scalar fields:')
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for r in a[-2:]:
 t=r['facility_telemetry_trace']; print(r['completed_campaign_short_name'],list(t) if isinstance(t,dict) else type(t).__name__)
 if isinstance(t,dict):print({k:v for k,v in t.items() if isinstance(v,(str,int,float,type(None)))})
