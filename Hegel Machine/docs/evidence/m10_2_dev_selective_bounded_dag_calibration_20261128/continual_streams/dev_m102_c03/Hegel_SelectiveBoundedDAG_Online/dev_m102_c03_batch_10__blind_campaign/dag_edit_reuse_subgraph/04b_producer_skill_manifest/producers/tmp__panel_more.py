import pickle,numpy as np
from scipy.optimize import minimize_scalar
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['panel_c']; u=p['u']; v=p['v1']
fs={'atan':lambda x,f:np.arctan(f*x),'asinh':lambda x,f:np.arcsinh(f*x),'tanh':lambda x,f:np.tanh(f*x),'sin':lambda x,f:np.sin(f*x),'softsign':lambda x,f:x/(1+f*abs(x)),'rat':lambda x,f:x/(1+f*x*x),'saturating':lambda x,f:x/np.sqrt(1+f*x*x),'gaussian':lambda x,f:x*np.exp(-f*x*x)}
for k,g in fs.items():
 for n in (False,True):
  def score(f):
   h=np.c_[np.ones(len(u)),g(u,f)]; h=np.c_[h,u] if n else h
   r=h@np.linalg.lstsq(h,v,rcond=None)[0]-v
   return np.sqrt(np.mean(r*r))
  opt=minimize_scalar(score,bounds=(.03,5),method='bounded')
  print(k,n,round(opt.x,6),round(opt.fun,7),round(score(1.25),7),round(score(2.15),7))
