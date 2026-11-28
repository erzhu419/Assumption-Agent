import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; x,y=s['ref_rosette'],s['raw_rosette'];ok=np.max(np.abs(y),axis=1)<8;x,y=x[ok],y[ok];b=s['single_bank_sweeps'];k,t,z=[b[a] for a in ('bank_id','ref_signal','raw_readout')];ok=np.abs(z)<8;k,t,z=k[ok],t[ok],z[ok];opp=(np.arange(4)+2)%4
forms={'atan':lambda x,a:np.arctan(a*x)/a,'tanh':lambda x,a:np.tanh(a*x)/a,'asinh':lambda x,a:np.arcsinh(a*x)/a,'sqrt':lambda x,a:x/np.sqrt(1+(a*x)**2),'ratio':lambda x,a:x/(1+(a*x)**2),'erf':None};from scipy.special import erf
forms['erf']=lambda x,a:erf(a*x)*np.sqrt(np.pi)/(2*a)
for key,f in forms.items():
 def res(p):return p[k]+p[4+k]*f(t,p[8])-z
 r=least_squares(res,np.r_[np.zeros(4),np.ones(4),1.],bounds=(np.r_[np.full(8,-np.inf),.01],np.inf),gtol=1e-14); print(key,'bank rms',np.std(r.fun),'params',r.x.round(8))
 if np.std(r.fun)<1e-5:
  p=r.x; v=x[:,opp]; c=(y-p[:4])/p[4:8]-f(x,p[8]); print('cross samples',np.array([v.ravel(),c.ravel()]).T[np.argsort(v.ravel())[::max(1,v.size//19)]].round(7));
  for name,fn in {**forms,'sin':lambda v,a:np.sin(a*v)/a,'linear':lambda v,a:v,'quadratic':lambda v,a:v*v,'cos':lambda v,a:1-np.cos(a*v),'exp':lambda v,a:np.expm1(a*v),'sigm':lambda v,a:np.tanh(a*v/2)}.items():
   for extra in (0,1):
    r2=least_squares(lambda q:(q[0]*fn(v,q[1])+extra*q[2]*v-c).ravel(),[.1,1.,0.],bounds=([-np.inf,.01,-np.inf],[np.inf,10,np.inf]),gtol=1e-13); print(name,extra,np.std(r2.fun),r2.x.round(8))
