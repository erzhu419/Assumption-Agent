import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement as cr
def load(f):
 s=importlib.util.spec_from_file_location('m',f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('/holdout_task/node_01.py'); upstream=load('/holdout_task/_frozen_nodes/node_00.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; n=len(y); q=m._calibrate(s['panel_a'],upstream.run_node({'__input__':s}),t['x']); b=m._panel_b(s['panel_b'],q); h=np.c_[np.ones(n),b]; ls=m._ls
outer=[np.arange(j,j+18) for j in (0,18,36,54)]+[np.arange(52,72),np.arange(0,72,4),np.argsort(np.max(abs(q),axis=1))[-15:],np.argsort(abs(q[:,0]**2-q[:,2]**2))[-15:]]
quad=np.c_[h,*[b[:,i]*b[:,j] for i,j in cr(range(4),2)]]
for mode in ('partial','offset','both','compact','mixed','cubic_gate'):
 for form in ('fixed','ratio','shift','cos','sum'):
  def design(a):
   x,z=q[:,0],q[:,2]; k=a[0]; r=a[1] if len(a)>1 else 1.
   arg= x*x-r*z*z if form in ('fixed','ratio') else (x*x-r*z*z-a[2] if form=='shift' else (np.cos(2.6*x)-r*np.cos(2.6*z) if form=='cos' else x*x+r*z*z))
   g=1/(1+k*arg**2)
   if mode=='offset':return np.c_[np.ones(n),quad*g[:,None]]
   if mode=='both':return np.c_[quad,quad*g[:,None]]
   if mode=='compact':return np.c_[h,h*g[:,None]]
   if mode=='mixed':return np.c_[h,quad*g[:,None]]
   if mode=='cubic_gate':return np.c_[quad,np.power(g[:,None],.5)*h]
   return np.c_[quad,g[:,None]*h]
  out=[]
  for j in outer+[np.array([],int)]:
   i=np.setdiff1d(np.arange(n),j)
   def res(a):
    f=design(a)[i];return f@ls(f,y[i])-y[i]
   start=[2.] if form=='fixed' else ([2.,1.,0.] if form=='shift' else [2.,1.]); hi=[15.,2.,.6][:len(start)]; lo=[.01,.3,-.6][:len(start)]; fit=least_squares(res,start,bounds=(lo,hi),max_nfev=100); f=design(fit.x); out.append((np.mean(abs(f[j]@ls(f[i],y[i])-y[j])) if j.size else np.sqrt(np.mean(fit.fun**2)),np.round(fit.x,3).tolist()))
  print(mode,form,np.round([v[0] for v in out],5),out[-1][1])
