import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for key in ['panel_b','panel_c']:
 p=s[key]; x=p['u']; n,d=x.shape
 for vk,y in p.items():
  if vk=='u':continue
  def affine(z):return np.c_[np.ones(n),z]
  def score(z):
   a=affine(z); return np.mean((a@np.linalg.lstsq(a,y,rcond=None)[0]-y)**2)**.5
  for name,fn in [('cos',np.cos),('sin',np.sin),('sin2',lambda z:np.sin(z)**2),('tanh',np.tanh),('both',lambda z:np.c_[np.cos(z),np.sin(z)]),('linear+cos',lambda z:np.c_[x,np.cos(z)]),('linear+sin',lambda z:np.c_[x,np.sin(z)])]:
   fs=np.linspace(.1,12,240); j=np.argmin([score(fn(f*x)) for f in fs]); opt=minimize_scalar(lambda f:score(fn(f*x)),bounds=(fs[max(j-1,0)],fs[min(j+1,len(fs)-1)]),method='bounded'); print(key,vk,name,round(opt.x,6),round(opt.fun,8), 'at2.15',round(score(fn(2.15*x)),8))
  print('poly',[(j,score(np.hstack([x**k for k in range(1,j+1)]))) for j in [1,2,3,4,5]])
