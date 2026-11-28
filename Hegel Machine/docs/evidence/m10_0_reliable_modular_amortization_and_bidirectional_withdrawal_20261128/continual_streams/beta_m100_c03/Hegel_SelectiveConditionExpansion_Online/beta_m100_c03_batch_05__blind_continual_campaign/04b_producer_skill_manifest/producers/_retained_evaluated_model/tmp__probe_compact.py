import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];c=s['sensor_calibration_sweeps'];d=c['single_bank_sweeps'];x=c['ref_rosette'];y=c['raw_rosette'];g=np.max(abs(y),axis=1)<6;b=d['bank_id'];t=d['ref_signal'];r=d['raw_readout'];sel=abs(r)<6
X=[np.r_[x[g,k],t[sel&(b==k)]] for k in range(4)]; V=[np.r_[x[g,(k+2)%4],np.zeros(np.sum(sel&(b==k)))] for k in range(4)];Y=[np.r_[y[g,k],r[sel&(b==k)]] for k in range(4)]
def diagonal(u,a,n):
 return (u+a[0]*u**3+a[1]*u**5)/(1+a[2]*u**2+a[3]*u**4+(a[4]*u**6 if n==5 else 0))
laws=[('linear+evenrat',lambda v,p:[v,v*v/(1+p[0]*v*v)],[.8]),('odd+evenrat',lambda v,p:[v/(1+p[0]*v*v),v*v/(1+p[1]*v*v)],[.05,.8]),('linear+odd+evenrat',lambda v,p:[v,v**3/(1+p[0]*v*v),v*v/(1+p[1]*v*v)],[.2,.8]),('tanh+evenrat',lambda v,p:[np.tanh(p[0]*v),v*v/(1+p[1]*v*v)],[.2,.8]),('rat2',lambda v,p:[v/(1+p[0]*v*v+p[1]*v**4),v*v/(1+p[0]*v*v+p[1]*v**4),v**3/(1+p[0]*v*v+p[1]*v**4),v**4/(1+p[0]*v*v+p[1]*v**4)],[.8,.02]),('cheb7',lambda v,p:[v**i for i in range(1,8)],[])]
for n in [4,5]:
 for label,law,p0 in laws:
  def fun(a,ret=False):
   errors=[];coef=[]
   for u,v,z in zip(X,V,Y):
    h=np.column_stack([np.ones(len(u)),diagonal(u,a,n),*law(v,a[n:])]);q=np.linalg.lstsq(h,z,rcond=None)[0];errors.extend(h@q-z);coef.append(q)
   return (np.array(errors),np.array(coef)) if ret else errors
  q0=[-.17,.061,.23,.028]+([.002] if n==5 else [])+p0
  lo=[-.45,0.,0.,0.]+([0.] if n==5 else [])+[1e-6]*len(p0)
  hi=[.5,.5,2.,2.]+([1.] if n==5 else [])+[5.]*len(p0)
  fit=least_squares(fun,q0,bounds=(lo,hi),max_nfev=150,gtol=1e-13)
  e,w=fun(fit.x,True);print(n,label,'rmse',np.sqrt(np.mean(e**2)),'max',max(abs(e)),'shape',fit.x.round(7),'coeff',w.round(6).tolist())
