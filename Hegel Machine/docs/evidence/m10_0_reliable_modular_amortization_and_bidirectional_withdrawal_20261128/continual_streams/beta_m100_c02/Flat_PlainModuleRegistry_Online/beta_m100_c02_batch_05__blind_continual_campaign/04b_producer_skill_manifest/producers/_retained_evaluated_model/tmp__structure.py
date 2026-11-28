import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; ok=np.max(abs(y),axis=1)<5
b=c['single_bank_sweeps']; u,v,j=(b[k] for k in ('ref_signal','raw_readout','bank_id')); B=np.eye(4)[j]; print('bank input grid',np.unique(u)); fold=np.random.default_rng(8).permutation(len(x))%5
# Nonlinear parameters are shared; gains and offsets are profiled independently.
forms={
'lin_sin':(lambda t,p:t+p[0]*np.sin(p[1]*t),[.25,2.5],[-1,.2],[2,7]),
'rat_sin':(lambda t,p:t/(1+p[2]*t*t)+p[0]*np.sin(p[1]*t),[.25,2.5,.01],[-1,.2,-.08],[2,7,2]),
'cubic_sin':(lambda t,p:t+p[0]*np.sin(p[1]*t)+p[2]*t**3,[.25,2.5,0],[-1,.2,-1],[2,7,1]),
'sin2':(lambda t,p:t+p[0]*np.sin(p[1]*t)+p[2]*np.sin(p[3]*t),[.25,2.5,.005,4],[-1,.2,-1,.2],[2,7,1,9]),
'linear_tanh_sin':(lambda t,p:t+p[0]*np.tanh(p[1]*t)+p[2]*np.sin(p[3]*t),[.05,1,.25,2.5],[-1,.1,-1,.3],[2,5,2,7])}
good=np.abs(v)<5; u,v,j,B=u[good],v[good],j[good],B[good]
best=None
for name,(f,start,lo,hi) in forms.items():
 def res(p):
  D=np.c_[B,B*f(u,p)[:,None]]; return D@np.linalg.lstsq(D,v,rcond=None)[0]-v
 sol=least_squares(res,start,bounds=(lo,hi),gtol=1e-13,max_nfev=250); print(name,np.sqrt(np.mean(sol.fun**2)),sol.x)
 if best is None or np.linalg.norm(sol.fun)<best[0]:best=(np.linalg.norm(sol.fun),f,sol.x)
_,f,p=best; F=lambda a:f(a,p); opp=np.roll(x,2,axis=1); z=F(x); w=F(opp)
for mode in range(10):
 err=[]; coeff=[]
 for k in range(5):
  te=ok&(fold==k); tr=ok&~te; Y=[]
  for j in range(4):
   a,t=x[:,j],opp[:,j]; g,h=z[:,j],w[:,j]; base=[np.ones(len(a)),g,h,h*h]; deriv=(F(a+1e-5)-F(a-1e-5))/2e-5; ds=(F(t+1e-5)-F(t-1e-5))/2e-5
   more=[[],[g*h],[t,t*t,g*h],[h**3,g*h,g*h*h,g*g*h],[a*t,a*a*t,a*t*t,h**3],[deriv*h,deriv*h*h],[g*h,ds*a,ds*a*a],[h*h/(1+h*h),h/(1+h*h),g*h],[t,t*t,g*h,g*h*h,g*g*h],[h/(1+h*h),h*h/(1+h*h),g*h,g*h*h,g*g*h]][mode]
   D=np.column_stack(base+more); beta=np.linalg.lstsq(D[tr],y[tr,j],rcond=None)[0]; Y.append(D[te]@beta-y[te,j]); coeff.append(beta)
  err.append(np.sqrt(np.mean(np.array(Y)**2)))
 print('pair',mode,round(np.mean(err),8),round(max(err),8),'coef bank0',np.round(coeff[0],5))
