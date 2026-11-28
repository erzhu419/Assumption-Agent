from wall_probe import load, axis
import itertools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
s,w,y=load()
nx,ny=s['nx'],s['ny']; rows=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]; E=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@w

def make(j, kind):
 k,v=s['kappa_'+j],s['vel_'+j]; n=len(k); h=s['L'+j]/(n-1); r=s['robin_'+j]; a=axis(k,v,h,r,[1,1,0,0]).tolil()
 if kind=='open': return axis(k,v,h,r,[1,1,1,1])
 if kind=='double': return axis(k,v,h,r,[2,2,0,0])
 if kind=='sealed': return a.tocsc()
 for i,inside,out,sign in [(0,1,n-1,1),(n-1,n-2,0,-1)]:
  fi=.5*(k[i]+k[inside])/h**2; fo=.5*(k[i]+k[out])/h**2
  if kind=='periodic': a[i,inside]=-fi-max(-sign*v[i],0)/h; a[i,out]=-fo-max(sign*v[i],0)/h
  if kind=='periodic_noadv': a[i,inside]=-fi; a[i,out]=-fo
  if kind=='periodic_local': a[i,inside]=-k[i]/h**2-max(-sign*v[i],0)/h; a[i,out]=-k[i]/h**2-max(sign*v[i],0)/h
  if kind=='periodic_local_noadv': a[i,inside]=-k[i]/h**2; a[i,out]=-k[i]/h**2
  if kind=='sealed_face': a[i,inside]=-fi
  if kind=='sealed_upwind': a[i,inside]=-fi-max(-sign*v[i],0)/h
  a[i,i]=-a[i,inside]-a[i,out]
 return a.tocsc()

def calc(ax,ay):
 L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+V
 return np.vstack([splu(L+t*E).solve(rhs)[rows] for t in s['mu_shifts']])

kinds='open double sealed periodic periodic_noadv periodic_local periodic_local_noadv sealed_face sealed_upwind'.split(); axes=[{k:make(j,k) for k in kinds} for j in 'xy']; scores=[]
for x,yy in itertools.product(kinds,repeat=2):
 e=np.linalg.norm(calc(axes[0][x],axes[1][yy])-y,axis=0)/np.linalg.norm(y,axis=0); scores.append((max(e),x,yy,e))
for row in sorted(scores)[:15]:print(row)
