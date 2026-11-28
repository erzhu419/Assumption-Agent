import numpy as np,pickle
from scipy.optimize import least_squares
x,y=[pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps'][k] for k in ('ref_rosette','raw_rosette')]
def pred(p,x,j,m):
 a=x[:,j]; h=p[2]+p[3]*a; lin=p[0]+x@p[4:8];
 if m in ('mix','mix_no_lin','mix_same'): h=h+x@p[8:12]
 if m=='mix_no_lin': lin=p[0]
 if m=='mix_same': lin=p[0]+p[4]*h
 if m=='rational': return lin+p[1]*h/np.sqrt(1+h*h)
 if m=='frac': return lin+p[1]*h/(1+h*h)
 if m=='poly5': return lin+sum(p[8+k]*a**(k+2) for k in range(4))
 if m=='affine': return lin
 return lin+p[1]*np.tanh(h)
def fit(x,y,m):
 b=np.column_stack((np.ones(len(x)),x)); w=np.ones(len(x),bool)
 for _ in range(4):
  r=np.linalg.norm(b@np.linalg.lstsq(b[w],y[w],rcond=None)[0]-y,axis=1); w=r<max(.06,4*np.median(r))
 ps=[]
 for j in range(4):
  p=np.zeros(12); p[1]=1.; p[3]=1.; p[j+4]=.25
  z=least_squares(lambda p:pred(p,x[w],j,m)-y[w,j],p,loss='soft_l1',f_scale=.015,max_nfev=400).x; ps.append(z)
 return np.array(ps),w
def predict(ps,x,m):return np.column_stack([pred(p,x,j,m) for j,p in enumerate(ps)])
if __name__=='__main__':
 for m in ('affine','poly5','rational','frac','own','mix','mix_no_lin','mix_same'):
  ps,w=fit(x,y,m); rp=y-predict(ps,x,m); folds=np.array_split(np.argsort(np.linalg.norm(x,axis=1)),5); errs=[]
  for ids in folds:
   tr=np.ones(len(x),bool);tr[ids]=False; p,_=fit(x[tr],y[tr],m); v=ids[w[ids]]; errs.append(np.sqrt(np.mean((y[v]-predict(p,x[v],m))**2)))
  print(m,'in',np.sqrt(np.mean(rp[w]**2,axis=0)).round(6),'OOF',np.round(errs,6));
  if m in ('own','mix','mix_same','mix_no_lin'): print(ps.round(5))
