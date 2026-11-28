import numpy as np,pickle
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=map(tr.get,('x','y','traj')); ts=s['eval_times']
np.set_printoptions(precision=4,suppress=True,linewidth=150)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in cr(range(4),k)])
def caldesign(u,j,n):
 a,b,c,d=np.roll(u,-j,axis=1).T
 return [aff(u),np.c_[np.ones(len(u)),a,b,a*b,a**3],np.c_[aff(u),u**2,u**3,a*b,a*c,a*d],poly(u,2),np.c_[aff(u),poly(u,2)[:,5:],u**3],np.c_[aff(u),u**2,u**3]][n]
def cal(n):
 u,v=map(s['panel_a'].get,('u','v')); A=[caldesign(u,j,n) for j in range(4)]; keep=np.ones(len(u),bool)
 for i in range(10):
  w=[ls(a[keep],v[keep,j]) for j,a in enumerate(A)]; r=np.linalg.norm(np.column_stack([a@b for a,b in zip(A,w)])-v,axis=1); keep=r<=np.sort(r)[36] if i<3 else r<max(.012,5*np.median(r))
 return w,r,keep
if __name__=='__main__':
 print('times telemetry spectrum',ts,s['telemetry_vec'],np.linalg.svd(T,compute_uv=False)); print('y/T corr',np.corrcoef(np.c_[y,T].T)[0]);print('raw extremes',x.min(0),x.max(0))
 for n in range(6):
  w,r,k=cal(n); print('CAL',n,k.sum(),np.quantile(r,[0,.5,.75,.8,1]));print(np.array(w))
 for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[p]['u'],s[p][k]; print(p,k)
  def f(u,j,mode):
   a,b,c,d=np.roll(u,-j,axis=1).T
   return [aff(u),poly(u,2),np.c_[aff(u),u*u],np.c_[aff(u),a*b,b*c,c*d,d*a,a*c,b*d],np.c_[aff(u),u**3],np.c_[aff(u),u**2,u**3]][mode]
  for mode in range(6):
   for name,g in [('id',lambda q:q),('sin',np.sin),('tanh',np.tanh),('soft',lambda q:q/np.sqrt(1+q*q)),('soft2',lambda q:q/np.sqrt(1+.1*q*q))]:
    vals=[]
    for j in range(4):
     A=f(u,j,mode); w=ls(A,v[:,j]); o=least_squares(lambda w:g(A@w)-v[:,j],w,max_nfev=100); Q=A*ls(A,A) .diagonal() if False else o.jac; h=np.sum(np.linalg.qr(Q,mode='reduced')[0]**2,axis=1); vals.append([np.sqrt(np.mean(o.fun**2)),np.mean(abs(o.fun/np.maximum(.02,1-h)))])
    print(mode,name,np.mean(vals,axis=0))
