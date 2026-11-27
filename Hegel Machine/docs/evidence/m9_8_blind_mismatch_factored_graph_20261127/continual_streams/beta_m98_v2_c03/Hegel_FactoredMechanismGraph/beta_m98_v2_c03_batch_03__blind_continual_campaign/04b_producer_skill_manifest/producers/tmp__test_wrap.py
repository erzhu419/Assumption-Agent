from fit_wall import *
from itertools import product

def axis(k,n,kind):
 h=s['L'+k]/(n-1); kap=s['kappa_'+k]; vel=s['vel_'+k]; rob=s['robin_'+k]
 if kind in ('open','reflect'):
  return m._axis(kap,vel,h,rob if kind=='open' else 0)
 a=m._axis(kap,vel,h,0).tolil()
 for j, inside, opposite in [(0,1,n-1),(n-1,n-2,0)]:
  d=kap[j]/h**2; f=(kap[j]+kap[inside])/(2*h**2); wrap=(kap[j]+kap[opposite])/(2*h**2)
  if kind=='periodic':
   vi=(max(-vel[j],0) if j==0 else max(vel[j],0))/h
   vw=(max(vel[j],0) if j==0 else max(-vel[j],0))/h
   a[j,inside]=-f-vi; a[j,opposite]=-wrap-vw; a[j,j]=f+vi+wrap+vw
  elif kind=='diffusive_wrap':
   a[j,opposite]=-wrap; a[j,j]=d+wrap
  elif kind=='mirror':
   a[j,inside]=-2*f; a[j,j]=2*f
  elif kind=='wall_upwind':
   vi=(max(-vel[j],0) if j==0 else max(vel[j],0))/h
   a[j,inside]=-f-vi; a[j,j]=f+vi
 return a.tocsc()

def eval_axes(ax,ay):
 L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.vstack([splu(L+t*I).solve(F)[sensor] for t in s['mu_shifts']])

rmse=lambda e:np.sqrt(np.mean(e**2))
kinds=('open','reflect','periodic','diffusive_wrap','mirror','wall_upwind')
results=[]
for kx,ky in product(kinds,repeat=2):
 e=eval_axes(axis('x',nx,kx),axis('y',ny,ky))-Y
 results.append((rmse(e),np.max(np.sqrt(np.mean(e.reshape(-1,ns,2)**2,axis=1))),kx,ky))
print('best discrete stencils (pooled, worst pulse-shift, x, y):'); print(*sorted(results)[:12],sep='\n')

def fitted_axes(p, leak):
 ans=[]
 for a0, (q,r), vs in zip(axes,base,p.reshape(2,2,-1)):
  a=a0.tolil(); n=a.shape[0]
  for i,(j,inside,opposite) in enumerate([(0,1,n-1),(n-1,n-2,0)]):
   f,w=vs[i,:2]*q[i]; l=vs[i,2]*r if leak else 0
   a[j,inside]=-f; a[j,opposite]=-w; a[j,j]=f+w+l
  ans.append(a.tocsc())
 return ans
for leak in (False,True):
 init=np.tile([1.,1.,0.] if leak else [1.,1.],4)
 lo=np.tile([0.,0.,-.2] if leak else [0.,0.],4)
 hi=np.tile([8.,8.,3.] if leak else [8.,8.],4)
 for label,keep in [('all',slice(None)),('pulse0',slice(0,1)),('pulse1',slice(1,2))]:
  def residual(p): return eval_axes(*fitted_axes(p,leak))-Y
  fit=least_squares(lambda p:residual(p)[:,keep].ravel(),init,bounds=(lo,hi),max_nfev=100,ftol=1e-11,gtol=1e-12,xtol=1e-11)
  e=residual(fit.x); print('fit wrap',leak,label,fit.x.round(6),'pulse RMSE',np.sqrt(np.mean(e*e,axis=0)),flush=True)
print('study keys',list(s)); print('grid and wall parameters',nx,ny,s['robin_x'],s['robin_y']); print('sensor on wall counts',[(ix==j).sum() for j in (0,nx-1)],[(iy==j).sum() for j in (0,ny-1)])
