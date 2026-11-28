import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
ls=lambda X,y:np.linalg.lstsq(X,y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def calibration(p):
 u,v=p['u'],p['v']; q=.18*np.tanh(1.8*u)-.32*np.sin(2.4*np.roll(u,2,axis=1)); Y=v-q; B=np.stack([ls(aff(u[:,j:j+1]),Y[:,j]) for j in range(4)],1)
 for _ in range(5):
  r=Y-B[0]-u*B[1]; m=np.max(abs(r),1)<.5; B=np.stack([ls(aff(u[m,j:j+1]),Y[m,j]) for j in range(4)],1)
 def f(x):return B[0]+B[1]*x+.18*np.tanh(1.8*x)-.32*np.sin(2.4*np.roll(x,2,axis=1))
 def inv(x):
  z=(x-B[0])/B[1]
  for _ in range(180):z=(x-B[0]-.18*np.tanh(1.8*z)+.32*np.sin(2.4*np.roll(z,2,axis=1)))/B[1]
  return z
 return f,inv
f,_=calibration(s['panel_a'])
def inv(x):
 z=x.copy()
 for _ in range(35):
  e=f(z)-x; d=(f(z+1e-5)-f(z-1e-5))/(2e-5) # only use finite differences per coordinate below
  J=np.stack([(f(z+1e-5*q)-f(z-1e-5*q))/(2e-5) for q in np.eye(4)],-1)
  delta=np.linalg.solve(J,e[...,None])[...,0]; z-=np.clip(delta,-.28,.28)
 return z
x=s['train']['x']; V=np.c_[s['train']['y'],s['train']['traj']]; z=inv(x); print('inverse residual',np.max(abs(f(z)-x)), 'range',z.min(),z.max())
from sklearn.preprocessing import PolynomialFeatures
ops={}
for key,col,nam,th in [('panel_b','v','b',[2.8478,1.5688,.39752]),('panel_c','v1','c',[2.43053,.00054,6]),('panel_c','v2','d',[1.2663,.00545,-.66712])]:
 u=s[key]['u'];v=s[key][col]; g=np.arctan if nam=='c' else np.cos
 def A(u,g=g,th=th):return aff(u+th[2]*g(th[0]*u+th[1]))
 w=ls(A(u),v); ops[nam]=lambda q,A=A,w=w:A(q)@w
 e=v-A(u)@w
 print(nam,'residual svd',np.linalg.svd(e,compute_uv=False)); print('correlations with residual',np.round(np.corrcoef(np.c_[e,u,u*u].T)[:4,4:],2))
coords={'z':z,'x':x,'fx':f(x)}
for _ in range(2):
 for nm,q in list(coords.items()):
  if len(nm.split('/'))!=_+1:continue
  for op,g in ops.items():coords[nm+'/'+op]=g(q)
rows=[]
for nm,q in coords.items():
 for deg in (1,2,3):
  D=PolynomialFeatures(deg).fit_transform(q); std=np.maximum(D[:52].std(0),.01); std[0]=1;D=D/std
  for a in (0,.01,.1,1):
   E=D@ls(np.r_[D[:52],np.sqrt(a)*np.eye(D.shape[1])],np.r_[V[:52],np.zeros((D.shape[1],9))])-V; rows.append((np.mean(abs(E[52:,0])),np.mean(abs(E[52:,1:])),nm,deg,a,np.mean(abs(E[:52,0])),np.mean(abs(E[:52,1:]))))
print('best y',sorted(rows)[:10]);print('best state',sorted(rows,key=lambda q:q[1])[:10]);print('times',s['eval_times']);print('svd traj',np.linalg.svd(V[:,1:],compute_uv=False))
