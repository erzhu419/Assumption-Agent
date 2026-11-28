import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
ls=lambda A,V:np.linalg.lstsq(A,V,rcond=None)[0]
def measure(X,V):
 B=ls(X,V); e=X@B-V; h=np.sum(X*np.linalg.pinv(X).T,1); return np.mean(abs(e)),np.mean(abs(e)/np.maximum(1-h[:,None],1e-8))
def aff(u):return np.c_[np.ones(len(u)),u]
funs={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'id':lambda x:x}
a=s['panel_a']; u=np.delete(a['u'],[5,17,31],0); v=np.delete(a['v'],[5,17,31],0); p=np.roll(u,2,axis=1)
res=[]
for name,f in funs.items():
 for name2,g in funs.items():
  for lin in [0,1]:
   def calc(z,ret=False):
    X=[np.column_stack([np.ones(len(u)),u[:,j],f(z[0]*u[:,j]),g(z[1]*p[:,j])]+([p[:,j]] if lin else [])) for j in range(4)]; B=[ls(A,v[:,j]) for j,A in enumerate(X)]; e=np.stack([A@b-v[:,j] for j,(A,b) in enumerate(zip(X,B))],1); return (B,e) if ret else e.ravel()
   opt=least_squares(calc,[1.25,2.15],bounds=([.25,.25],[5,5])); B,e=calc(opt.x,True);res.append((np.mean(abs(e)), name,name2,lin,opt.x,np.array(B)))
for r in sorted(res,key=lambda x:x[0])[:9]:print('A',r)
for pk,vk in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pk]['u']; v=s[pk][vk]; X=aff(u); W=ls(X,v); res=[]
 for nam,f in funs.items():
  if nam=='id':continue
  for shape in [0,1,2]:
   for off in [False,True]:
    def calc(z,ret=False):
     if shape==0: A=aff(f(z[0]*u+z[1]*off));
     elif shape==1: A=np.c_[X,f(z[0]*u+z[1]*off)];
     else: A=aff(u+z[0]*f(z[1]*u+z[2]*off));
     return measure(A,v) if ret else (A@ls(A,v)-v).ravel()
    for z in ([1.25,.1],[2.15,0]) if shape<2 else ([.22,1.25,.1],[1.,2.15,0]):
     opt=least_squares(calc,z,max_nfev=100,bounds=(-5,5)); res.append((calc(opt.x,True),nam,shape,off,opt.x))
 print(pk,vk,'elementwise',sorted(res,key=lambda x:x[0][1])[:7]); print('pure square',measure(np.c_[X,u*u],v));
 for j in range(4):
  res=[]
  for name,f in funs.items():
   if name=='id':continue
   for shape in [0,1,2]:
    def calc(z):
     r=X@z[:5]; A=np.c_[np.ones(len(u)), f(r)] if shape==0 else (np.c_[X,f(r)] if shape==1 else np.c_[np.ones(len(u)),r,f(r)]); return A@ls(A,v[:,j])-v[:,j]
    for k in [0.5,1.25,2.15,3.5]:
     opt=least_squares(calc,W[:,j]*k,max_nfev=140);res.append((np.mean(abs(opt.fun)),name,shape,opt.x))
  print('ridge',j,sorted(res,key=lambda x:x[0])[:3])
