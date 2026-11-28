import numpy as np
from scipy.optimize import least_squares
from operators import pa,poly

def reverse_a(p,x):
 u,v=p['u'],p['v']; m=pa(p); u,v=u[m],v[m]; j=[1,2,3,0]
 b=np.stack([np.linalg.lstsq(np.c_[np.ones(len(u)),u[:,i],u[:,k],u[:,i]*u[:,k],u[:,i]**3],v[:,i],rcond=None)[0] for i,k in enumerate(j)],axis=1)
 z=(x-b[0])/b[1]
 for _ in range(20):
  f=b[0]+b[1]*z+b[2]*z[:,j]+b[3]*z*z[:,j]+b[4]*z**3-x
  J=np.zeros((len(x),4,4)); i=np.arange(4); J[:,i,i]=b[1]+b[3]*z[:,j]+3*b[4]*z*z; J[:,i,j]=b[2]+b[3]*z
  z-=np.linalg.solve(J,f[...,None])[...,0]
 return z

def cx(p):
 u=p['u']; P=poly(u,1); Q=poly(u,2)[:,5:]; from itertools import combinations_with_replacement
 ij=list(combinations_with_replacement(range(4),2))
 ops=[]
 for label in ['v1','v2']:
  vs=p[label]; out=[]
  for v in vs.T:
   base=np.linalg.lstsq(P,v,rcond=None)[0]; q=np.linalg.lstsq(np.c_[P,Q],v,rcond=None)[0][5:]; M=np.zeros((4,4));
   for c,(i,j) in zip(q,ij): M[i,j]=M[j,i]=c/(2 if i!=j else 1)
   ev,E=np.linalg.eigh(M); k=np.argmax(abs(ev)); dd=E[:,k]; print(label,'quadratic eig',ev.round(4),'linear',base[1:].round(4))
   if label=='v1':
    pred=lambda h,x:h[0]+x@h[1:5]+h[5]*(x@h[1:5])**3
    r=least_squares(lambda h:pred(h,u)-v,np.r_[base,.15]); print('cubic ridge',r.x.round(5),np.std(r.fun)); out.append((pred,r.x))
   else:
    pred=lambda h,x:h[0]+x@h[1:5]+h[9]*(x@h[5:9])**2
    r=least_squares(lambda h:pred(h,u)-v,np.r_[base,dd,ev[k]]); print('quadratic ridge',r.x.round(5),np.std(r.fun)); out.append((pred,r.x))
  ops.append(lambda x,out=out:np.column_stack([f(h,x) for f,h in out]))
 return ops

if __name__=='__main__':
 import pickle
 from importlib.machinery import SourceFileLoader
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; z=reverse_a(s['panel_a'],x); print('latent ranges',z.min(0),z.max(0)); p=s['panel_b']; b=np.linalg.lstsq(poly(p['u'],1),p['v'],rcond=None)[0]; print('B singular',np.linalg.svd(b[1:],compute_uv=False)); cb=lambda x:poly(x,1)@b; c1,c2=cx(s['panel_c']); mod=SourceFileLoader('prior','/tmp/hegel_continual_archive/conf_m101_b03_batch_00_solver.py').load_module(); H,m=mod._time_modes(s['eval_times'],tr['traj']); Y=np.c_[tr['y'],m];
 for tag,w in [('x',x),('z',z),('b(x)',cb(x)),('b(z)',cb(z)),('b^-1(z)',(z-b[0])@np.linalg.inv(b[1:])),('c1(z)',c1(z)),('c2(z)',c2(z)),('c1(b(z))',c1(cb(z))),('c2(b(z))',c2(cb(z)))]:
  print('coordinates',tag)
  for label,D in [('lin',poly(w,1)),('quadratic',poly(w,2)),('cubic',poly(w,3)),('trig',np.c_[poly(w,2),np.sin(w),np.cos(w)]),('sat',np.c_[poly(w,2),np.tanh(w)])]:
   B=np.linalg.lstsq(D[:52],Y[:52],rcond=None)[0]; R=D@B-Y; print(label,'fit/cv',np.mean(abs(R[:52]),axis=0).round(5),np.r_[np.mean(abs(R[52:,0])),np.mean(abs(R[52:,1:]@H.T))].round(5))
   if label=='lin': print('coef', B.round(3).T)
