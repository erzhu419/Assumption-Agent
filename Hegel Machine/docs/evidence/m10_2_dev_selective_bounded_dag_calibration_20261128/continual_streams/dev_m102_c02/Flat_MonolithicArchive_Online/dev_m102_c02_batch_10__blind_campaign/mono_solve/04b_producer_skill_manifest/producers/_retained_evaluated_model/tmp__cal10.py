import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.linalg.norm(v,axis=1)<4;u=u[keep];v=v[keep]
def ls(h,y):return np.linalg.lstsq(h,y,rcond=None)[0]
def fit(h,v=v):
 c=np.stack([ls(h[:,j],v[:,j]) for j in range(4)]);return np.einsum('ijk,jk->ij',h,c)-v,c
forms={ 'linear':(lambda x,p:x,[]), 'tanh':(lambda x,p:np.tanh(x*p[0]),[2]), 'sin':(lambda x,p:np.sin(x*p[0]),[2]), 'atan':(lambda x,p:np.arctan(x*p[0]),[2]),'erf':(lambda x,p:erf(x*p[0]),[2]),'power':(lambda x,p:np.sign(x)*abs(x)**p[0],[1.5]),'rat':(lambda x,p:x/(1+p[0]*abs(x)**p[1]),[.3,2]),'sat':(lambda x,p:x/np.sqrt(1+p[0]*abs(x)**p[1]),[.22,2.15]),'odd':(lambda x,p:np.sign(x)*abs(x)**p[0]/(1+p[1]*abs(x)**p[2]),[1,.3,2]) }
def explore():
 out=[]
 for k,(f,p) in forms.items():
  for l,(g,q) in forms.items():
   def make(x,a):return np.stack([np.ones_like(x),f(x,a[:len(p)]),g(np.roll(x,2,axis=1),a[len(p):])],axis=2)
   def err(a):return fit(make(u,a))[0].ravel()
   start=p+q; a=least_squares(err,start,bounds=(.0001,40),max_nfev=200).x if start else []; out.append((np.std(err(a)),k,l,a,fit(make(u,a))[1].T))
 for row in sorted(out,key=lambda x:x[0])[:18]:print(row)
 h=np.stack([np.ones_like(u),u,u*u,u**3,u**5,np.roll(u,2,axis=1),np.roll(u,2,axis=1)**2,np.roll(u,2,axis=1)**3,np.roll(u,2,axis=1)**5,u*np.roll(u,2,axis=1)],2);print('poly',np.std(fit(h)[0]),fit(h)[1].T);print('panel ranges',u.min(0),u.max(0))
if __name__=='__main__':explore()
