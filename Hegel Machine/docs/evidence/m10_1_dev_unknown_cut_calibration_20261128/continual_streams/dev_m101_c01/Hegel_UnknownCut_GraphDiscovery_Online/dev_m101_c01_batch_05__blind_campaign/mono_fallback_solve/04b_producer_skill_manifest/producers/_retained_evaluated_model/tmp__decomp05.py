exec(open('/tmp/probe05.py').read().split('p=temp(z[:56])')[0])
from scipy.linalg import svd
u=s['panel_c']['u']; f=np.c_[np.ones(len(u)),u]; B=np.linalg.lstsq(f,np.c_[s['panel_c']['v1'],s['panel_c']['v2']],rcond=None)[0]; r=np.c_[s['panel_c']['v1'],s['panel_c']['v2']]-f@B
print('panel C v1,v2 residual singular values',*(np.linalg.svd(v,compute_uv=False) for v in np.split(r,2,axis=1)))
print('panel u range',u.min(0),u.max(0))
for name,F in [('diagonal',np.c_[f,u*u]),('diagonal+cube',np.c_[f,u*u,u**3]),('trig',np.c_[f,np.sin(1.7*u),np.cos(1.7*u)]),('diagonal B',np.c_[f,b(u)**2]),('diagonal+trig',np.c_[f,u*u,np.sin(1.7*u),np.cos(1.7*u)])]:
 U,S,V=svd(F,full_matrices=False); E=r-U@(U.T@r); print(name,'loo rms',np.sqrt(np.mean((E/(1-(U*U).sum(1))[:,None])**2,axis=0)))
p=temp(z[:56]); R=z-amp(z,p)@basis(p).T; U,S,V=svd(R,full_matrices=False);print('trajectory residual singular',S, 'modes',V[:3]);print('residual std by time',R.std(0)); print('actual traj range',z.min(0),z.max(0))
