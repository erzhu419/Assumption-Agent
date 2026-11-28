from panel12 import *
from scipy.optimize import least_squares
from itertools import product
np.set_printoptions(precision=4,suppress=True)
ps={k:s['panel_'+k] for k in 'abc'}
for p,key in [('a','v'),('b','v'),('c','v1'),('c','v2')]:
 q=ps[p]; print('\nPANEL',p,key,flush=True)
 r=fitpanel(q['u'],q[key],p=='a'); print([i[:4] for i in r if not i[2].startswith('cube')][:8],flush=True)
 if p=='a':
  u,v=q['u'],q[key]; a=aff(u); keep=np.arange(len(u));
  for it in range(9):
   w=ls(a[keep],v[keep]); err=np.linalg.norm(a@w-v,axis=1); keep=np.argsort(err)[:37]
  print('a row residuals',np.sort(err)); print('a coef',w); print('a delta patterns',np.c_[u,v-u][keep[:20]])
 else:
  u,v=q['u'],q[key]; a=aff(u); gate=r[0][0]
  for typ in ['sin','cos','tanh','asinh','soft','rat','sig']:
   fun=forms[typ][0]; rms=[]; fits=[]
   for j in range(4):
    y=v[:,j]; base=ls(a,y); rng=np.random.default_rng(98+j)
    def res(p): return p[5]+p[6]*fun(a@p[:5])-y
    starts=[np.r_[base*k,0,1./k] for k in [.5,1,2,3,5]]+[np.r_[rng.normal(0,2,5),0,1] for i in range(20)]
    opt=min((least_squares(res,p,max_nfev=160) for p in starts),key=lambda f:f.cost); rms.append(np.sqrt(np.mean(opt.fun**2))); fits.append(opt.x)
   print('outer',typ,rms, np.array(fits) if max(rms)<.004 else '',flush=True)
