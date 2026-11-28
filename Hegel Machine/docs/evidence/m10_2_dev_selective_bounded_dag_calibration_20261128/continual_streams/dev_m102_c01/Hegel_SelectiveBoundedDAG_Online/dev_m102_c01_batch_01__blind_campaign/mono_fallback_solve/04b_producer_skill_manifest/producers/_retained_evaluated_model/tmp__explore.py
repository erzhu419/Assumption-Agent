import numpy as np,pickle
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; n=len(x)
np.set_printoptions(precision=5,suppress=True,linewidth=130)
def design(u): return np.c_[np.ones(len(u)),u]
def fit(f,v): return np.linalg.lstsq(f,v,rcond=None)[0]
a=s['panel_a']; u=a['u']; v=a['v']; keep=np.ones(len(u),bool)
for _ in range(5):
 A=fit(design(u[keep]),v[keep]); e=np.linalg.norm(design(u)@A-v,axis=1); keep=e<max(.002,4*np.median(e));
print('a',A,'a residuals',np.sort(e))
b=s['panel_b']; B=fit(design(b['u']),b['v']);
c=s['panel_c']; u=c['u']; C=[]
for name in ['v1','v2']:
 v=c[name]; opts=[]
 for kind in range(7):
  def f(u,k):
   if kind==0: return design(np.tanh(k*u))
   if kind==1: return np.c_[design(u),u*u/(1+k*u*u)]
   if kind==2: return design(np.sin(k*u))
   if kind==3: return np.c_[design(u),u*u]
   if kind==4: return design(u/(1+k*u*u))
   if kind==5: return np.c_[design(u), np.roll(u,1,axis=1)*u]
   if kind==6: return np.c_[design(u),np.cos(k*u)]
  def loss(k):
   F=f(u,k); Q=np.linalg.qr(F)[0]; return np.mean(((v-F@fit(F,v))/(1-(Q**2).sum(1)[:,None]))**2)**.5
  k=minimize_scalar(loss,bounds=(.001,2.4),method='bounded').x; opts.append((loss(k),kind,k));
 print(name, sorted(opts));
 kind=min(opts)[1]; k=min(opts)[2]; W=fit(f(u,k),v); print('W',W)
 if kind==0: C.append(lambda z,k=k,W=W: design(np.tanh(k*z))@W)
 elif kind==1: C.append(lambda z,k=k,W=W: np.c_[design(z),z*z/(1+k*z*z)]@W)
 else: C.append(lambda z:z)
print('eval times',t, 'telemetry',s['telemetry_vec']); U,S,V=np.linalg.svd(T,full_matrices=False); print('trajectory sv',S,'modes',V[:3]); print('y on traj',np.mean(abs(T@fit(T,y)-y)))
z=(x-A[0])@np.linalg.inv(A[1:]); bases={'x':x,'z':z,'Bx':design(x)@B,'Bz':design(z)@B}; bases.update({f'C{i+1}{k}':func(q) for k,q in list(bases.items()) for i,func in enumerate(C)});
for key,q in bases.items():
 i,j=np.triu_indices(4); F=np.c_[design(q),q[:,i]*q[:,j]]; p=F@fit(F[:52],np.c_[y,T][:52]); print(key,'lin coeff',fit(design(q)[:52],T[:52,-1]),'quad CV',np.mean(abs(p[52:]-np.c_[y,T][52:]),axis=0)[[0,1,8]], 'train',np.mean(abs(p[:52]-np.c_[y,T][:52]),axis=0)[[0,1,8]])
