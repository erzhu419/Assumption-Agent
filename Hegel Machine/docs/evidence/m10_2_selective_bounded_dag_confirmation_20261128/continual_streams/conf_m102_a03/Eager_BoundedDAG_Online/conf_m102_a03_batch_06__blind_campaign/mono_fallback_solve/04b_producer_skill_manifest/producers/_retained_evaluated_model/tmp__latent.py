import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=170)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def inv(x,c):
 z=np.linalg.solve(np.diag(c[1])+np.roll(np.diag(c[2]),1,axis=1),(x-c[0]).T).T
 for k in range(16):
  n=np.roll(z,-1,axis=1);pred=c[0]+c[1]*z+c[2]*n+c[3]*z*n
  D=np.eye(4)[None,:,:]*(c[1]+c[3]*n)[:,:,None]+np.roll(np.eye(4),1,axis=1)[None,:,:]*(c[2]+c[3]*z)[:,:,None]
  step=np.linalg.solve(D,(x-pred)[...,None])[...,0]; z+=np.clip(step,-1,1)
 return z
u=s['panel_a']['u'];v=s['panel_a']['v'];n=np.roll(u,-1,axis=1);A=np.stack([u*0+1,u,n,u*n],axis=2);w=np.ones(len(u));
for i in range(40):
 c=np.stack([ls(A[:,j]*w[:,None],v[:,j]*w) for j in range(4)],axis=1);r=np.linalg.norm(np.einsum('ijk,kj->ij',A,c)-v,axis=1);w=np.sqrt(np.minimum(1,1.3*np.median(r)/np.maximum(r,1e-12)))
g=r<4*np.median(r);c=np.stack([ls(A[g,j],v[g,j]) for j in range(4)],axis=1); x=s['train']['x'];z=inv(x,c);y=s['train']['y'];t=s['train']['traj'];P=np.c_[y,t];print('cal',c,'z range',z.min(0),z.max(0));print('inverse test/train bounds',inv(s['test']['x'],c).min(0),inv(s['test']['x'],c).max(0));
Q,S,V=np.linalg.svd(t,full_matrices=False);print('time spectrum',S,'modes',V[:3]);print('trajectory ratios',V[0]/V[0,-1]);print('feature coeff',ls(aff(z),np.c_[y,Q[:,:3]*S[:3]]));
def check(F,P=P):
 er=np.abs(F[52:]@ls(F[:52],P[:52])-P[52:]);return (er.mean(0)[0],er.mean(0)[1:].mean())
for m,X in [('x',x),('cal',z)]:
 for deg in range(1,4):
  p=PolynomialFeatures(deg); F=p.fit_transform(X);print(m,deg,check(F));
  if deg==2:
   print(p.get_feature_names_out(),ls(F[:52],P[:52])[:,[0,1,-1]].T)
 for nm,h in [('tanh',np.tanh),('sin',np.sin),('exp',np.exp),('abs',abs),('rat',lambda z:z/(1+.22*z*z))]:
  for mult in [.5,1,1.25,2]:
   F=PolynomialFeatures(2).fit_transform(h(X*mult));print(m,nm,mult,check(F))
for pn in ['panel_b','panel_c']:
 p=s[pn];u=p['u'];a=aff(u)
 for vn,v in p.items():
  if vn=='u':continue
  B=ls(a,v);E=v-a@B;print(pn,vn,'sv resid',np.linalg.svd(E,compute_uv=False),'coeff',B)
  for f in [lambda x:x,np.tanh,np.sin,lambda x:x/(1+.22*x*x)]:
   Z=f(aff(z)@B);print('op',check(PolynomialFeatures(2).fit_transform(Z)))
print('x z y end first16\n',np.c_[x,z,y,t[:,-1]][:16])
