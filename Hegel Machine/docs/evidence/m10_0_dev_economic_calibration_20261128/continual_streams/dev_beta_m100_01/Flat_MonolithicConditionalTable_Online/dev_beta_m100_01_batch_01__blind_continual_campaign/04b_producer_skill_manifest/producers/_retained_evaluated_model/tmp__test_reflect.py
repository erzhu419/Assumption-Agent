from fit_wall import *
from itertools import product

def op(kind, q=(1.,1.,1.,1.), robin=(0.,0.,0.,0.)):
 axes=[]
 for a,n,idx in [('x',nx,0),('y',ny,2)]:
  h=s['L'+a]/(n-1); k=s['kappa_'+a]; v=s['vel_'+a]; r=s['robin_'+a]; A=m._axis(k,v,h,r).tolil()
  for t,(i,j) in enumerate([(0,1),(n-1,n-2)]):
   base=k[i]/h**2; mid=(k[i]+k[j])/(2*h*h); adv=max((i-j)*v[i],0)/h
   c={'local':base,'face':mid,'double_local':2*base,'double_face':2*mid,'face_upwind':mid+adv,'local_upwind':base+adv,'double_face_upwind':2*mid+adv,'double_local_upwind':2*base+adv,'local_absadv':base+abs(v[i])/h,'face_absadv':mid+abs(v[i])/h}[kind]
   A[i,i]=q[idx+t]*c+robin[idx+t]*r/h; A[i,j]=-q[idx+t]*c
  axes.append(A.tocsc())
 return sparse.kron(sparse.eye(ny),axes[0],format='csc')+sparse.kron(axes[1],sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')

def err(B): return W@m._forward(s,B)[0].T-y

def stats(e): return np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e.reshape(3,3,-1)**2,axis=2)).max()

for k in ('local','face','double_local','double_face','face_upwind','local_upwind','double_face_upwind','double_local_upwind','local_absadv','face_absadv'):
 print(k,stats(err(op(k))))
for shared in (True,False):
 for held in (None,0,1,2):
  fun=lambda p:err(op('local',tuple(np.repeat(p,4) if shared else p)))
  train=np.ones(3,dtype=bool) if held is None else ix!=held
  p=least_squares(lambda p:fun(p)[train].ravel(),np.ones(1 if shared else 4),gtol=1e-13).x
  e=fun(p); print('fit reflecting',shared,held,p,stats(e),None if held is None else np.sqrt(np.mean(e[held].reshape(3,-1)**2,1)))
print('boundary kappa:',s['kappa_x'][[0,1,-2,-1]], s['kappa_y'][[0,1,-2,-1]])
print('boundary velocities:',s['vel_x'][[0,-1]],s['vel_y'][[0,-1]])
