from fit_dag05 import *
from scipy.linalg import qr
from scipy.optimize import least_squares

def bank(x):
 F={}
 for i in range(4): F[f'x{i}']=x[:,i]
 for j in (2,3):
  for ix in itertools.combinations_with_replacement(range(4),j): F['p'+''.join(map(str,ix))]=np.prod(x[:,ix],axis=1)
 for i in range(4):
  for j in range(i+1,4):
   for s in (-1,1):F[f'x{i}{s:+}x{j}']=x[:,i]+s*x[:,j]
 for name,q in list(F.items()):
  if name.startswith('p') and len(name)>3:continue
  for f,fn in [('sin',np.sin),('s2',lambda v:np.sin(2*v)),('s3',lambda v:np.sin(3*v)),('tanh',np.tanh),('cos',np.cos),('rat',lambda v:v/(1+v*v)),('abs',np.abs)]:F[f'{f}({name})']=fn(q)
 return F

def greedy(x,Y,n,base=1):
 F=bank(x); names=list(F); C=np.array(list(F.values())).T; A=poly(x,base)
 for k in range(n):
  Q=qr(A,mode='economic')[0]; R=Y-Q@(Q.T@Y); D=C-Q@(Q.T@C)
  norms=np.linalg.norm(D,axis=0); c=np.linalg.norm(D.T@R,axis=-1) if R.ndim==2 else abs(D.T@R)
  j=np.argmax(np.where(norms>1e-7,c/np.maximum(norms,1e-7),0)); A=np.column_stack((A,C[:,j])); print(names[j],score(A,Y))
 return A
pc=raw['panel_c']; u=pc['u']; v1=pc['v1']; v2=pc['v2']; P=poly(u,2); print('v2 quad coefficients',np.round(np.linalg.lstsq(P,v2,rcond=None)[0],3)); print('v1 lin coefficients',np.round(np.linalg.lstsq(poly(u,1),v1,rcond=None)[0],3)); print('v1 greedy'); greedy(u,v1,8); print('v2 greedy');greedy(u,v2,10)
print('scalar greedy'); greedy(z,y,14)
print('traj greedy');greedy(z,T[:,-1],12)
print('mean shape',np.linalg.svd(T,full_matrices=False)[2][:2]); print('scalar plus trajectory modes',score(np.column_stack((poly(z,1),np.linalg.svd(T,full_matrices=False)[0][:,:2])),y))
