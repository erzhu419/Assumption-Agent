import pickle, numpy as np
from scipy.optimize import minimize_scalar, least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); good=np.max(abs(v-u),axis=1)<2; u,v=u[good],v[good]; n,d=u.shape

def fit(a,y):
 c=np.linalg.lstsq(a,y,rcond=None)[0]; r=y-a@c; q=np.linalg.qr(a,mode='reduced')[0]; return np.mean(r*r),np.mean((r/np.maximum(1-(q*q).sum(1),1e-8))**2),c

def design(u,j,f,kind,mode,i=None):
 q=u[:,j]; h=u[:,(j+1)%4]; z=u[:,(j+2)%4 if i is None else i]; a=np.c_[np.ones(len(u)),u,q*h,q**3]
 if mode==1: a=np.c_[a,q*q]
 if mode==2: a=np.c_[np.ones(len(u)),u,u*u,u**3,*[u[:,k]*u[:,l] for k,l in combinations(range(4),2)]]
 if mode==3: a=np.c_[a,q*q,h*h,q*q*h,q*h*h]
 term={'sin':lambda:np.sin(f*z),'tanh':lambda:np.tanh(f*z),'rat':lambda:z/(1+f*z*z),'sc':lambda:np.c_[np.sin(f*z),np.cos(f*z)],'atan':lambda:np.arctan(f*z)}[kind]()
 return np.c_[a,term]

def search(j,kind,mode,i):
 b=(.05,5.5) if kind in ('sin','sc') else (.03,6); fs=np.linspace(*b,90); y=v[:,j]; obj=lambda f: fit(design(u,j,f,kind,mode,i),y)[0]; k=np.argmin([obj(f) for f in fs]); f=minimize_scalar(obj,bounds=(fs[max(0,k-1)],fs[min(len(fs)-1,k+1)]),method='bounded').x; a,b,c=fit(design(u,j,f,kind,mode,i),y); return (b**.5,a**.5,kind,mode,i,f,c)

if __name__=='__main__':
 print('reference rows',len(u))
 for j in range(4):
  allr=[search(j,k,m,i) for i in range(4) for m in [0,1,2,3] for k in ['sin','tanh','rat','sc','atan']]; print('j',j); print('best',[(tuple(np.round(x,7) if isinstance(x,float) else x for x in r[:-1]),r[-1].round(5)) for r in sorted(allr,key=lambda r:r[0])[:6]]); print('simple',[(tuple(np.round(x,7) if isinstance(x,float) else x for x in r[:-1]),r[-1].round(5)) for r in sorted((r for r in allr if r[3]==0),key=lambda r:r[0])[:4]])
 for mode in [0,1,2,3]:
  for k in ['sin','tanh','rat','sc','atan']:
   obj=lambda f:sum(fit(design(u,j,f,k,mode),v[:,j])[0] for j in range(4)); f=minimize_scalar(obj,bounds=(.05,5),method='bounded').x; print('shared',mode,k,round(f,5),np.array([fit(design(u,j,f,k,mode),v[:,j])[:2] for j in range(4)]).__pow__(.5).round(6).T.tolist())
