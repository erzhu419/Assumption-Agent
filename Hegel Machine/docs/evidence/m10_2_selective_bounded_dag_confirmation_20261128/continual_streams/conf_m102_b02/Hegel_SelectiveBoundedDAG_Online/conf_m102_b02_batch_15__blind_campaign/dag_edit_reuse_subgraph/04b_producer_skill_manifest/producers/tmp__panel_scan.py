import pickle,numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for p,k in [('a','v'),('b','v'),('c','v1'),('c','v2')]:
 u,v=[s['panel_'+p][i] for i in ('u',k)]; n,d=u.shape; one=np.ones((n,1)); keep=np.ones(n,bool)
 if p=='a':
  for j in range(4):
   a=np.column_stack([one,u]); r=np.abs(a@np.linalg.lstsq(a,v[:,j],rcond=None)[0]-v[:,j]); print('A raw',j,np.round(r,2))
  for _ in range(8):
   a=np.column_stack((one,u));r=np.linalg.norm(a@np.linalg.lstsq(a[keep],v[keep],rcond=None)[0]-v,axis=1); keep=r<min(2.,4*np.median(r))
 x=u[keep];y=v[keep]; print(p,k,'kept',sum(keep)); out=[]
 for name,f in [('poly2',lambda x,t:x*x),('poly3',lambda x,t:x**3),('abs',lambda x,t:abs(x)),('signed-square',lambda x,t:x*abs(x)),('tanh',lambda x,t:np.tanh(t*x)),('erf',lambda x,t:erf(t*x)),('sin',lambda x,t:np.sin(t*x)),('cos',lambda x,t:np.cos(t*x)),('exp-square',lambda x,t:np.exp(-t*x*x)),('rational-square',lambda x,t:x*x/(1+t*x*x)),('rational-odd',lambda x,t:x/(1+t*x*x)),('rational-abs-square',lambda x,t:x*x/(1+t*abs(x))),('arcsinh',lambda x,t:np.arcsinh(t*x)),('softsign',lambda x,t:x/(1+t*abs(x)))]:
  for linear in [False,True]:
   def calc(t,loo=True):
    h=np.column_stack([np.ones(len(x)),*([x] if linear else []),f(x,t)]); q=np.linalg.qr(h)[0];r=y-q@(q.T@y); return np.sqrt(np.mean((r/(1-(q*q).sum(1))[:,None] if loo else r)**2))
   ts=np.r_[np.geomspace(.04,4,45),np.linspace(4.2,8,20)]; i=np.argmin([calc(t) for t in ts]); a,b=ts[max(i-1,0)],ts[min(i+1,len(ts)-1)]; t=minimize_scalar(calc,bounds=(a,b),method='bounded').x;out.append((round(calc(t),6),name,linear,round(t,4),round(calc(t,False),6)))
 print(*sorted(out)[:12],sep='\n')
 if p=='a':
  for lag in range(1,4):
   for z in [x*np.roll(x,lag,1),np.roll(x,lag,1)**2,x**3]:
    h=np.column_stack([np.ones(len(x)),x,z]); r=y-h@np.linalg.lstsq(h,y,rcond=None)[0]; print('A interaction',lag,np.sqrt(np.mean(r*r,axis=0)))
