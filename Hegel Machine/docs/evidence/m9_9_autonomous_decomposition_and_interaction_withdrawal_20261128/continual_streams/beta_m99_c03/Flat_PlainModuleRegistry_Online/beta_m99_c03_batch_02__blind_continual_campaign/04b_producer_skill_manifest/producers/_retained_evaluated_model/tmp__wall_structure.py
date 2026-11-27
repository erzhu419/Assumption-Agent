from boundary_fit import *
np.set_printoptions(precision=8)
start=np.tile([1.,1.,1.,1.],2)
def params(t):
 return np.r_[0.,0.,t[:4],0.,0.,t[4:]]
for leave in [None,0,1,2]:
 train=[j for j in range(W.shape[1]) if j!=leave]
 f=least_squares(lambda t:res(params(t))[:,train].ravel(),start,ftol=1e-13,xtol=1e-13,gtol=1e-13)
 print('zero-leak held pulse',leave,'factors',f.x,'pulse RMSE',np.mean(res(params(f.x))**2,axis=0)**.5)
 if leave is None: full=f.x
for i,a in enumerate('xy'):
 k,v=S['kappa_'+a],S['vel_'+a];h=S['L'+a]/(len(k)-1)
 t=full[4*i:4*i+4]; local=k[[0,-1]]; faces=(local+k[[1,-2]])/2; wrap=local.mean(); vel=v[[0,-1]]*h; direction=np.array([1.,-1.]); inward=np.maximum(-direction*vel,0); outward=np.maximum(direction*vel,0)
 print(a,'h',h,'local',local,'inward faces',faces,'wrap face',wrap,'velocity*h',vel)
 print('inferred inward',t[:2]*local,'inferred wrap',t[2:]*faces)
 for base in [local,faces,np.full(2,wrap)]:
  print('base',base,'plus inward',base+inward,'plus outward',base+outward)
# Fully periodic row-local FVM: averaged neighbor faces and upwind on both wall rows.
periodic=[]
for a in 'xy':
 k,v=S['kappa_'+a],S['vel_'+a]; h=S['L'+a]/(len(k)-1); loc=k[[0,-1]]; face=(loc+k[[1,-2]])/2; d=h*v[[0,-1]]*np.array([1.,-1.]); periodic.extend(np.r_[(face+np.maximum(-d,0))/loc,(loc.mean()+np.maximum(d,0))/face])
r=res(params(periodic)); print('periodic upwind factors',periodic,'pulse RMSE',np.mean(r*r,axis=0)**.5)
print('pulse weight singular values',np.linalg.svd(W,compute_uv=False))
