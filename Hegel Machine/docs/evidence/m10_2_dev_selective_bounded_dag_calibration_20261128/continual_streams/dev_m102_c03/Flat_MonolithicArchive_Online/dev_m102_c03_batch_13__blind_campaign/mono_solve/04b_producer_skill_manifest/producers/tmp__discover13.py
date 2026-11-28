import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
u,v=[s['panel_a'][k] for k in ['u','v']]
def cal(z,j,style):
 w=z[:,j]; q=np.roll(z,-j,axis=1); p=q[:,1]
 opts=[aff(z), np.c_[np.ones(len(z)),w,p,w*p,w**3],np.c_[aff(z),q[:,0,None]*q[:,1:],w**3],np.c_[aff(z),q[:,0,None]*q[:,1:],q**2,q**3],np.c_[aff(z),q[:,0,None]*q[:,1:],w**3,np.sin(2.4*q[:,2])]]
 return opts[style]
for st in range(5):
 keep=np.linalg.norm(v-aff(u)@ls(aff(u),v),axis=1)<2.5
 for it in range(5):
  A=[cal(u,j,st) for j in range(4)]; W=[ls(a[keep],v[keep,j]) for j,a in enumerate(A)]
  r=np.linalg.norm(np.column_stack([a@w for a,w in zip(A,W)])-v,axis=1); keep=r<max(4*np.median(r),.001)
 print('A',st,np.round(np.sort(r),4),[w.round(4).tolist() for w in W] if st in (1,4) else '')
funcs={'soft':lambda z,r:z/(1+r*abs(z)), 'rat':lambda z,r:z/(1+(r*z)**2),'gauss':lambda z,r:np.exp(-(r*z)**2),'tanh':lambda z,r:np.tanh(r*z),'sin':lambda z,r:np.sin(r*z),'cos':lambda z,r:np.cos(r*z),'cosh':lambda z,r:1/(1+(r*z)**2),'abs':lambda z,r:abs(z),'sq':lambda z,r:z*z,'cube':lambda z,r:z**3,'prod':lambda z,r:z*np.roll(z,-1,axis=1),'prod2':lambda z,r:z*np.roll(z,-2,axis=1)}
for k in ['v1','v2']:
 u,v=[s['panel_c'][i] for i in ['u',k]]; print(k)
 for name,f in funcs.items():
  opt=[]
  for linear in [False,True]:
   def des(z,r):return np.c_[aff(z) if linear else np.ones((len(z),1)),f(z,r)]
   def res(r):
    a=des(u,r); return (a@ls(a,v)-v).ravel()
   r=min([least_squares(res,np.ones(4)*i,bounds=(.02,6),max_nfev=100) for i in [.5,1.6,2.5,4]],key=lambda p:np.linalg.norm(p.fun)).x
   a=des(u,r); h=(a*np.linalg.pinv(a).T).sum(axis=1); e=a@ls(a,v)-v
   opt.append((np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/np.maximum(.005,1-h[:,None]))**2)),r.round(4)))
  print(name,opt)
t=s['eval_times']; Y=s['train']['traj']
def modes(r):return -np.expm1(-t[:,None]*r)
r=least_squares(lambda r:(modes(r)@ls(modes(r),Y.T)-Y.T).ravel(),[.4,1.85]).x
print('time rates',r,'rmse',np.sqrt(np.mean((modes(r)@ls(modes(r),Y.T)-Y.T)**2)))
