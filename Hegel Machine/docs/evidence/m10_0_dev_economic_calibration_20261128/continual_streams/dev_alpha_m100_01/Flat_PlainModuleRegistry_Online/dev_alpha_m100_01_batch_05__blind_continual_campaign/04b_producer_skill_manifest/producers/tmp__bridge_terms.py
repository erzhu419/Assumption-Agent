import numpy as np,pickle
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; clean=np.max(abs(y),axis=1)<5
terms={
 'tanh_shift': (lambda v,k,s:np.tanh(k*v+s)-np.tanh(s), [.5,.6]),
 'atan_shift': (lambda v,k,s:np.arctan(k*v+s)-np.arctan(s), [.5,.6]),
 'exp': (lambda v,k:-np.expm1(-k*v), [.3]),
 'rat':(lambda v,k:v/(1+k*v), [.15]),
 'quadratic':(lambda v,k:v+k*v*v, [-.1]),
 'sin_cos':(lambda v,k,s:np.sin(k*v)+s*(np.cos(k*v)-1), [.6,.5]),
 'log':(lambda v,k:np.log1p(k*v), [.2]),
 'linear_tanh':(lambda v,k,s:v+s*(np.tanh(k*v+.6)-np.tanh(.6)), [.6,1]),
}
for j in range(4):
 u,v=x[:,j],x[:,j^2]; z=y[:,j]
 for name,(term,start) in terms.items():
  for interaction in ('none','uv','sum_inside','sum_difference'):
   def pred(p):
    b,a,k,d=p[:4]; m=0 if interaction=='none' else p[-1]; w=p[4:4+len(start)]
    if interaction=='sum_difference': return b+a*np.tanh(k*u)+d*(term(v+m*u,*w)-term(m*u,*w))
    return b+a*np.tanh(k*(u+(m*v if interaction=='sum_inside' else 0)))+d*term(v,*w)+(m*u*v if interaction=='uv' else 0)
   ini=[0,1.8 if j%2==0 else 1.1,1,.35]+start+([] if interaction=='none' else [0.]); lo=np.full(len(ini),-np.inf); hi=-lo
   if name in ('rat','log'):lo[4]=-.53;hi[4]=.53
   errs=[]
   for fold in range(5):
    val=clean&(np.arange(len(x))%5==fold); tr=clean&~val
    p=least_squares(lambda p:(pred(p)-z)[tr],ini,bounds=(lo,hi),max_nfev=180).x; errs.extend((pred(p)-z)[val])
   p=least_squares(lambda p:(pred(p)-z)[clean],ini,bounds=(lo,hi),max_nfev=260).x
   print(j,name,interaction,'%.6f %.6f %.6f'%(np.mean(np.square(errs))**.5,max(abs(np.array(errs))),np.mean((pred(p)-z)[clean]**2)**.5),np.round(p,4))
