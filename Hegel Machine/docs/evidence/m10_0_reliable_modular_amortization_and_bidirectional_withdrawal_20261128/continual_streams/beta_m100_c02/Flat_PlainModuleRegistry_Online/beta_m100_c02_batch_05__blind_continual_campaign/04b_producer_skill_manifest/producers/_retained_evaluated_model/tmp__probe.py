import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; ok=np.max(abs(y),axis=1)<5
print('excluded',np.flatnonzero(~ok)); b=c['single_bank_sweeps']; bx=b['ref_signal']; by=b['raw_readout']; bank=b['bank_id'];
forms={'affine':lambda x,a:x, 'tanh':lambda x,a:np.tanh(a*x)/a, 'atan':lambda x,a:np.arctan(a*x)/a, 'sqrt':lambda x,a:x/np.sqrt(1+a*x*x), 'rat':lambda x,a:x/(1+a*x*x), 'asinh':lambda x,a:np.arcsinh(a*x)/a}
for i in range(4):
 u=(bank==i)&(abs(by)<5); xx=bx[u]; yy=by[u];
 for name,f in forms.items():
  for extra in (0,1,2,3):
   def mat(a):
    z=f(xx,a[0]); return np.column_stack([np.ones(len(xx)),z]+([xx**2] if extra==1 else [z**2] if extra==2 else [xx] if extra==3 else []))
   def res(a):
    A=mat(a);return A@np.linalg.lstsq(A,yy,rcond=1e-12)[0]-yy
   sol=least_squares(res,[.6],bounds=(.005,10),gtol=1e-14); A=mat(sol.x); print(i,name,extra,'%.3g'%np.sqrt(np.mean(sol.fun**2)),np.round(sol.x,6),np.round(np.linalg.lstsq(A,yy,rcond=1e-12)[0],6))
print('cross validation on clean rosette rows'); rng=np.random.default_rng(8); fold=rng.permutation(len(x))%5
for name,fun in forms.items():
 for ext in (0,1,2,3):
  def design(x,a):
   z=fun(x,a); e=[x*x] if ext==1 else [z*z] if ext==2 else [x*x]+[(x[:,i]*x[:,j])[:,None] for i in range(4) for j in range(i+1,4)] if ext==3 else []; return np.column_stack([np.ones(len(x)),z]+e)
  err=[]
  for i in range(5):
   tr=ok&(fold!=i); te=ok&(fold==i)
   def coef(a):return np.linalg.lstsq(design(x[tr],a),y[tr],rcond=1e-12)[0]
   sol=least_squares(lambda a:(design(x[tr],a)@coef(a)-y[tr]).ravel(),np.full(4,.6),bounds=(.005,10),gtol=1e-12)
   err.append(np.sqrt(np.mean((design(x[te],sol.x)@coef(sol.x)-y[te])**2)))
  print(name,ext,'mean/worst',np.mean(err),max(err))
