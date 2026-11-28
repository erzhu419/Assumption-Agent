exec(open('/tmp/probe11.py').read().split('for k,p in')[0])
from scipy.optimize import least_squares
p=s['panel_a'];u=p['u'];v=p['v']; mask=np.max(abs(v),axis=1)<3;print('A clean',sum(mask),np.sort(np.max(abs(v),axis=1)))
for i in range(4):
 print('A coord',i)
 for label,a in [('lin',aff(u)),('quad',polys(u,2)),('cub',polys(u,3)),('owncub',np.c_[aff(u),u**3]),('cyclic',np.c_[aff(u),u[:,i]*u[:,(i+1)%4],u[:,i]**3]),('reverse',np.c_[aff(u),u[:,i]*u[:,(i-1)%4],u[:,i]**3])]:
  c=fit(a[mask],v[mask,i]); print(label,np.sqrt(np.mean((a[mask]@c-v[mask,i])**2)),np.round(c,3) if a.shape[1]<10 else '')
 for f in (np.tanh,np.sin,erf):
  for j in [i]:
   def ds(w):return np.c_[aff(u),f(w*u[:,j])]
   opt=minimize_scalar(lambda w:np.mean((ds(w)[mask]@fit(ds(w)[mask],v[mask,i])-v[mask,i])**2),bounds=(.1,4),method='bounded');print(f.__name__,opt.x,np.sqrt(opt.fun))
p=s['panel_c'];u=p['u'];v=p['v1'];r=p['v2'];print('C coef V2',np.round(fit(polys(u,2),r),4));print('C lin V1',np.round(fit(aff(u),v),4))
for f in (np.tanh,erf,np.sin):
 for i in range(4):
  def ds(w):return np.c_[aff(u),f(w*u[:,i])]
  opt=minimize_scalar(lambda w:np.mean((ds(w)@fit(ds(w),v[:,i])-v[:,i])**2),bounds=(.1,5),method='bounded');print('C',i,f.__name__,opt.x,np.sqrt(opt.fun),np.round(fit(ds(opt.x),v[:,i]),4))
for in_ in [v,u,np.c_[u,v]]:
 ev('C v2 residual on '+str(in_.shape),polys(in_,1),r) if False else print('C v2 on',in_.shape,[np.sqrt(np.mean((a@fit(a,r)-r)**2)) for a in (polys(in_,1),polys(in_,2))])
print('traj singulars',np.linalg.svd(t,compute_uv=False));print('y-from-traj')
for k in (1,2,3):
 U,S,V=np.linalg.svd(t[:n],full_matrices=False);w=t@V[:k].T
 for j in (1,2,3):ev(f'traj* {k} {j}',polys(w,j));ev(f'x+traj {k} {j}',polys(np.c_[x,w],j))
