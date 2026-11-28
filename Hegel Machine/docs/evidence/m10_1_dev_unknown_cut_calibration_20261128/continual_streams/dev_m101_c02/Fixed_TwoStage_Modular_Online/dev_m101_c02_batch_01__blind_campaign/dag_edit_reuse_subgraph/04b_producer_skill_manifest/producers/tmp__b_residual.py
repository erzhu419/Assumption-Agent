import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=s['panel_b']['u'],s['panel_b']['v'];n=len(u)
fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
forms={'sin':(lambda x,w:np.sin(x*w),2.8,(.3,6)), 'rational':(lambda x,w:x**3/(1+w*x*x),.75,(.01,6)), 'tanh':(lambda x,w:np.tanh(x*w),1.4,(.3,6)), 'softsign':(lambda x,w:x/np.sqrt(1+w*x*x),1.,(.01,8)), 'arctan':(lambda x,w:np.arctan(x*w),1.3,(.3,6)), 'erf_like':(lambda x,w:x*np.exp(-w*x*x),.7,(.01,6))}
pool={}
for deg in (2,3):
 for ix in combinations_with_replacement(range(4),deg): pool[str(ix)]=np.prod(u[:,ix],axis=1)
for i,j in combinations(range(4),2):
 for sign in (-1,1):
  z=u[:,i]+sign*u[:,j]
  for name,g in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos)]:
   for w in (.6,1.2,2.4):pool[(name,i,j,sign,w)]=g(w*z)
for name,(fun,start,bounds) in forms.items():
 def design(w):return np.c_[np.ones(n),u,fun(u,w)]
 def residual(w):
  d=design(w); return (v-d@fit(d,v)).ravel()
 opt=least_squares(residual,[start]*4,bounds=bounds); D=design(opt.x); R=residual(opt.x).reshape(v.shape); scores=[]
 for k,col in pool.items():
  r=col-D@fit(D,col);scores.append((np.linalg.norm(r@R)/max(np.linalg.norm(r),1e-10),str(k)))
 print(name,np.round(opt.x,4),'residual',np.linalg.norm(R),'top omitted',sorted(scores,reverse=True)[:7]);print('residual singular values',np.linalg.svd(R,compute_uv=False))
