import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rng=np.random.default_rng(19)
F={'cube':lambda z:z**3,'rat':lambda z:z/(1+z*z),'tanh':np.tanh,'sin':np.sin,'abs':np.abs,'rat2':lambda z:z*z/(1+z*z),'exp':lambda z:np.exp(-z*z),'relu':lambda z:np.maximum(z,0)}
for name,key in [('panel_b','v'),('panel_c','v1')]:
 x=s[name]['u']; v=s[name][key]; a=np.c_[np.ones(len(x)),x]; pa=np.linalg.pinv(a); dx=np.stack([x[:,j]+sign*x[:,k] for j,k in combinations(range(4),2) for sign in (1,-1)],axis=1); z=np.c_[x,dx]; p=lambda d:d-a@(pa@d)
 print(name,flush=True)
 for typ,f in F.items():
  opts=[]
  for form,d in [('diagonal',f(x)),('pairs',f(dx)),('mixed',f(z)),('squares+diag',np.c_[x*x,f(x)])]:
   h=np.c_[a,d]; r=v-h@np.linalg.lstsq(h,v,rcond=None)[0]; lev=np.sum(h*np.linalg.pinv(h).T,axis=1); opts.append((form,round(np.std(r),4),round(np.std(r/np.maximum(1-lev[:,None],1e-10)),4)))
  print(typ,opts,flush=True)
  for shared in (True,False):
   out=[]
   for yy in ([v] if shared else v.T[:,:,None]):
    Y=p(yy)
    def resid(w):
     h=p(f(a@w))[:,None]; h0=h[:,0]; return (h@(h.T@Y/max(h0@h0,1e-20))-Y).ravel()
    fits=[least_squares(resid,rng.normal(size=5)*k,bounds=(-10,10),max_nfev=100) for k in [.5,1,2] for _ in range(5)]
    o=min(fits,key=lambda o:o.cost);out.append((round(np.std(o.fun),4),np.round(o.x,2).tolist()))
   print('shared' if shared else 'individual',out,flush=True)
