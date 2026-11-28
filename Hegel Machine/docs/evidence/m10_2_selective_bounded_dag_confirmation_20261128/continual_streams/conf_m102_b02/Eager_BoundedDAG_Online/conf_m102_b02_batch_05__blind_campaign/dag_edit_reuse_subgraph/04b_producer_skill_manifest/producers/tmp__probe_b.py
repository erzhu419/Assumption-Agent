import sys,pickle,itertools,numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=map(s['panel_b'].get,('u','v'));print('u ranges',u.min(0),u.max(0));print('u/v rows',np.c_[u,v].round(4)); pairs=list(itertools.combinations(range(4),2))
extra={'none':lambda u:np.empty((len(u),0)), 'u':lambda u:u,'sq':lambda u:u*u,'cross':lambda u:np.column_stack([u[:,i]*u[:,j] for i,j in pairs]),'cubic':lambda u:u**3,'sin':lambda u:np.sin(u),'cos':lambda u:np.cos(u),'tanh':lambda u:np.tanh(u),'const_sum_sq':lambda u:(u*u).sum(1)[:,None],'sin_sum':lambda u:np.sin(u.sum(1))[:,None],'prod':lambda u:u.prod(1)[:,None]}
for name,fn in [('erf',erf),('tanh',np.tanh),('sin',np.sin),('rat',lambda t:t/(1+t*t)),('exp',lambda t:t*np.exp(-t*t)),('cos',np.cos),('gauss',lambda t:np.exp(-t*t))]:
 for ext,g in extra.items():
  def fit(q):
   F=np.c_[np.ones(len(u)),fn(q[0]*u)+q[1]*u,g(u)]; C=np.linalg.lstsq(F,v,rcond=1e-10)[0];return F,C,F@C-v
  sols=[least_squares(lambda q: fit(q)[2].ravel(),[a,-.6*a],bounds=([.05,-20],[8,20]),max_nfev=220) for a in (.5,1.4,2.5,4)]
  o=min(sols,key=lambda o:o.cost);F,C,R=fit(o.x);loo=R/(1-np.sum(F*np.linalg.pinv(F).T,axis=1))[:,None]; print(name,ext,o.x.round(5),'rms %.6f loo %.6f'%(np.mean(R**2)**.5,np.mean(loo**2)**.5));
  if name=='erf' and ext in ('none','u','sq','cross'): print('C',C.round(4).tolist())
