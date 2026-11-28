import numpy as np,pickle
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def one(*a):return np.column_stack([np.ones(len(a[0])),*a])
def ls(h,y):return np.linalg.lstsq(h,y,rcond=None)[0]
func={'sin':lambda u,k:np.sin(k*u),'cos':lambda u,k:np.cos(k*u)-1,'tanh':lambda u,k:np.tanh(k*u),'atan':lambda u,k:np.arctan(k*u),'gauss':lambda u,k:np.expm1(-k*u*u),'rat':lambda u,k:u/(1+k*u*u),'evenrat':lambda u,k:u*u/(1+k*u*u)}
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=140)
 for p,key in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u=s[p]['u']; v=s[p][key]; keep=np.linalg.norm(v,axis=1)<4; u=u[keep];v=v[keep]; h=one(u); print(p,key,'linear',ls(h,v))
  for name,f in func.items():
   def err(k):
    h=one(u,f(u,k)); r=v-h@ls(h,v); return np.mean(r*r)
   ks=np.linspace(.01,6,100); k0=ks[np.argmin([err(k) for k in ks])]; k=minimize_scalar(err,bounds=(max(.001,k0-.07),k0+.07),method='bounded').x
   print(name,k,np.sqrt(err(k)))
   if np.sqrt(err(k))<.003:print(ls(one(u,f(u,k)),v))
