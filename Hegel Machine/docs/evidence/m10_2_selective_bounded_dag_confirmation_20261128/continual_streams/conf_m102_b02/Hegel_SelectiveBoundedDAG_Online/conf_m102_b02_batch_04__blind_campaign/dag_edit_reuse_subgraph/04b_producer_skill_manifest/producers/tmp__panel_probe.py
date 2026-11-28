import pickle,numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def d(*xs): return np.column_stack((np.ones(len(xs[0])),*xs))
def fit(a,v):return np.linalg.lstsq(a,v,rcond=None)[0]
def err(a,v):
 r=a@fit(a,v)-v; h=np.sum(a*np.linalg.pinv(a).T,axis=1);return np.mean(abs(r)),np.mean(abs(r)/np.maximum(1e-9,1-h[:,None]))
funcs={'erf':erf,'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'asinh':np.arcsinh}
def bf(z,p,key,mode):
 h=funcs[key](np.exp(p[0])*z+(p[1]*z**3 if mode in (2,3,4) else 0))
 if mode==0:return d(h)
 if mode in (1,2):return d(z,h)
 if mode==3:return d(h+p[2]*z+p[3]*z**3)
 if mode==4:return d(z,z**3,h)
 if mode==5:return d(z,z**3,h)
 if mode==6:return d(z,z**3,z**5)
 if mode==7:return d(z/(1+np.exp(p[0])*z*z),z)
 if mode==8:return d(z/(1+np.exp(p[0])*z*z),z**3)
 if mode==9:return d(h,z*np.tanh(p[1]*z*z))

def bfit(u,v):
 out=[]
 for key in funcs:
  for mode in range(10):
   if mode in (6,7,8) and key!='erf':continue
   def r(p):
    a=bf(u,p,key,mode);return (a@fit(a,v)-v).ravel()
   best=None
   for z in ([.4,0.,0.,0.],[.5,2.,-2.,-.5],[-.2,-.4,.3,.1]):
    n={0:1,1:1,2:2,3:4,4:2,5:1,6:1,7:1,8:1,9:2}[mode]
    p=least_squares(r,z[:n],bounds=([-3]+[-12]*(n-1),[2]+[12]*(n-1)),max_nfev=150).x
    mse=np.mean(r(p)**2)
    if best is None or mse<best[0]:best=(mse,p)
   p=best[1]; a=bf(u,p,key,mode); out.append((err(a,v)[1],key,mode,p,fit(a,v),err(a,v)[0]))
 return sorted(out,key=lambda a:a[0])

def c2fit(u,v):
 out=[]
 for key,fun in [('rat2',lambda u,k:u*u/(1+k*u*u)),('rat1',lambda u,k:u*u/(1+k*abs(u))),('cos',lambda u,k:np.cos(k*u)),('atan',lambda u,k:np.arctan(k*u*u)),('abs',lambda u,k:abs(u)),('sqrt',lambda u,k:np.sqrt(1+k*u*u)),('exp',lambda u,k:np.exp(-k*u*u))]:
  for mode in (0,1,2):
   def f(z,p):
    q=fun(z,np.exp(p[0])); return d(z,q) if mode==0 else d(z,z*z,q) if mode==1 else d(z+p[1]*q)
   def r(p):
    a=f(u,p);return (a@fit(a,v)-v).ravel()
   p=least_squares(r,[-1.5] if mode!=2 else [-1.5,.45],bounds=([-4] if mode!=2 else [-4,-5],[2] if mode!=2 else [2,5])).x
   a=f(u,p);out.append((err(a,v)[1],key,mode,p,fit(a,v),err(a,v)[0]))
 return sorted(out,key=lambda a:a[0])

if __name__=='__main__':
 for o in bfit(s['panel_b']['u'],s['panel_b']['v'])[:18]:print('B',o[:4],o[-1],flush=True);print(np.round(o[4],5)) if o[0]<.002 else None
 for o in c2fit(s['panel_c']['u'],s['panel_c']['v2'])[:10]:print('C2',o[:4],o[-1],flush=True);print(np.round(o[4],5)) if o[0]<.0002 else None
