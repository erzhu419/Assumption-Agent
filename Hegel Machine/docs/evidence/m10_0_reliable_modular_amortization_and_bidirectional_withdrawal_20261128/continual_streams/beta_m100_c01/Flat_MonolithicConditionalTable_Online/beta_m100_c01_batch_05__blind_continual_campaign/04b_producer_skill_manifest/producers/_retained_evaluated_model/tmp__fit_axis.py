import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; X=sw['ref_rosette']; Y=sw['raw_rosette']; b=sw['single_bank_sweeps']; np.set_printoptions(precision=6,suppress=True)
axis=np.count_nonzero(X,axis=1)==1
print('axis rows',np.c_[X[axis],Y[axis]])
print('nonaxis examples',np.c_[X[~axis],Y[~axis]][:6]); print('archive wall axis');
fun={'tanh':lambda t,a:np.tanh(a*t), 'atan':lambda t,a:np.arctan(a*t), 'asinh':lambda t,a:np.arcsinh(a*t), 'erf':lambda t,a:erf(a*t), 'rational':lambda t,a:t/(1+a*t*t), 'sqrt':lambda t,a:t/np.sqrt(1+a*t*t), 'sine':lambda t,a:np.sin(a*t)}
from scipy.special import erf
for i in range(4):
 q=(b['bank_id']==i)&(abs(b['raw_readout'])<5); r=axis&(X[:,i]!=0)&(abs(Y).max(1)<5); x=np.r_[b['ref_signal'][q],X[r,i]]; y=np.r_[b['raw_readout'][q],Y[r,i]]
 for name,f in fun.items():
  for kind in ['linear','quadratic','cubic']:
   def pred(p): return p[0]+p[1]*f(x,p[2])+p[3]*x+(p[4]*(x*x if kind=='quadratic' else x**3) if kind!='linear' else 0)
   p=least_squares(lambda p:pred(p)-y,[0,1,1,.1,0][:4+(kind!='linear')],bounds=([-np.inf,-np.inf,0,-np.inf,-np.inf][:4+(kind!='linear')],np.inf),max_nfev=1000).x
   print(i,name,kind,'rmse',np.sqrt(np.mean((pred(p)-y)**2)),'p',p)
