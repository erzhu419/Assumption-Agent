import numpy as np,pickle,importlib.util
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=tr['traj']; y=tr['y']; t=s['eval_times']; n=52
A=s['panel_a']; u,v=A['u'],A['v']; ok=np.linalg.norm(v,axis=1)<5; u,v=u[ok],v[ok]; w=np.roll(u,2,axis=1)
def fit(h,y): return np.linalg.lstsq(h,y,rcond=1e-10)[0]
def aff(h): return np.c_[np.ones(len(h)),h]
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/dev_m102_c02_batch_09_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def cal(p,Z,k):
 W=np.roll(Z,2,axis=-1); a,b,c,d=p
 if k==0: return np.stack((np.ones_like(Z),Z/np.sqrt(1+a*abs(Z)**b),W/(1+c*abs(W)**d)),axis=-1)
 if k==1: return np.stack((np.ones_like(Z),Z,Z*W,W,np.sin(a*Z),np.sin(a*W)),axis=-1)
 return np.stack((np.ones_like(Z),Z/(1+a*abs(Z)**b)**c,np.sin(d*W)),axis=-1)
X={'old':m._calibrate(A,x)}
for k,init,lo,hi in [(0,[.5,2,.5,2],[.001,.3,.001,.3],[8,6,8,6]),(1,[2.4,2,2,2],[1,.1,.1,.1],[4,5,5,5]),(2,[.5,2,.5,2.4],[.001,.3,.01,.5],[8,6,8,6])]:
 def prof(p):
  h=cal(p,u,k); c=np.array([fit(h[:,j],v[:,j]) for j in range(4)]);return (h*c).sum(-1)-v,c
 p=least_squares(lambda p:prof(p)[0].ravel(),init,bounds=(lo,hi),gtol=1e-12).x; e,c=prof(p); print('cal',k,p,np.std(e),np.round(c,4)); X[str(k)]=np.array([least_squares(lambda z:(cal(p,z,k)*c).sum(-1)-q,q,gtol=1e-11).x for q in x])
P=s['panel_b']; B=lambda z:aff(np.cos(2.6*z))@fit(aff(np.cos(2.6*P['u'])),P['v']); E=-np.expm1(-t[:,None]*[.41318229,1.81372115]); T=fit(E,Y.T).T
for key,z in X.items():
 q=B(z); co=np.cos(2.6*z); Hs={}
 for label,a in [('b',q),('cos',co)]:
  Hs[label]=aff(a); Hs[label+'2']=aff(np.c_[a,*[a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(4),2)]]); Hs[label+'+sin']=aff(np.c_[a,np.sin(2.15*a),np.cos(2.15*a)]); Hs[label+'+tanh']=aff(np.c_[a,np.tanh(2.15*a)]); Hs[label+'+exp']=aff(np.c_[a,np.exp(-.22*a*a)]); Hs[label+'+rat']=aff(np.c_[a,a/(1+.22*a*a)]); Hs[label+'+z']=aff(np.c_[a,z]); Hs[label+'+soft']=aff(np.c_[a,np.logaddexp(0,2.15*a)]);
 for j,tar in enumerate([y,T[:,0],T[:,1],Y]):
  scores=[]
  for l,h in Hs.items():
   f=h@fit(h[:n],tar[:n]);scores.append((np.mean(abs(f[n:]-tar[n:])),np.mean(abs(f[:n]-tar[:n])),l))
  print(key,j,sorted(scores)[:5]);
 print('B polynomial coefs y',fit(Hs['b2'][:n],y[:n]));print('B poly coefs T',fit(Hs['b2'][:n],T[:n]).T)
np.savez('/tmp/ref15.npz',**X)
