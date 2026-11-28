import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps'];x=c['ref_rosette'];y=c['raw_rosette']; bank=c['single_bank_sweeps']; good=np.max(abs(y),axis=1)<6
fns={'tanh':lambda t,b:np.tanh(b*t),'atan':lambda t,b:np.arctan(b*t),'rational':lambda t,b:t/(1+b*t*t),'sqrt':lambda t,b:t/np.sqrt(1+b*t*t),'erf':lambda t,b:__import__('scipy').special.erf(b*t)}
for name,f in fns.items():
 def fit(t,v,linear):
  def basis(b):return np.column_stack([np.ones(len(t)),f(t,b[0])]+([t] if linear else []))
  def res(b):
   z=basis(b);return z@np.linalg.lstsq(z,v,rcond=None)[0]-v
  opt=least_squares(res,[1.],bounds=(.03,15),ftol=1e-12,gtol=1e-12,xtol=1e-12);z=basis(opt.x);return opt.x[0],np.linalg.lstsq(z,v,rcond=None)[0],np.sqrt(np.mean(opt.fun**2))
 for linear in (False,True):
  pars=[]
  for k in range(4):
   m=(bank['bank_id']==k)&(abs(bank['raw_readout'])<6);pars.append(fit(bank['ref_signal'][m],bank['raw_readout'][m],linear))
  print(name,linear,[(round(a,6),b.round(6).tolist(),round(e,7)) for a,b,e in pars])
  if name=='tanh' or name=='rational':
   a=np.array([p[0] for p in pars]); q=f(x,a);i,j=np.triu_indices(4,1)
   for n,t in [('q',[q,x*x]),('qx',[q,x,x*x]),('qpair',[q,x*x,x[:,i]*x[:,j]]),('qxpair',[q,x,x*x,x[:,i]*x[:,j]]),('qpairT',[q,x*x,q[:,i]*q[:,j]]),('qxcube',[q,x,x*x,x**3])]:
    z=np.column_stack([np.ones(len(x)),*t]); b=np.linalg.lstsq(z[good],y[good],rcond=None)[0];e=(z@b-y)[good];print(' ',n,np.sqrt(np.mean(e**2,0)).round(7),np.max(abs(e)))
    if n=='qxpair' and linear:print('  B',b.round(6).T)
