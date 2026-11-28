import pickle,runpy,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/conf_m101_b04_batch_05_solver.py'); ls=m['_ls']; aff=m['_affine']; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; q=s['train']; x=q['x']; y=q['y']; tr=q['traj']; z=m['_calibrate'](s['panel_a'],x); w=m['_panel_b'](s['panel_b'],z); np.set_printoptions(precision=5,suppress=True)
u,v=s['panel_a'].values(); uu=np.roll(u,-1,axis=1); a=[np.c_[np.ones(len(u)),b,c,b*c,b**3] for b,c in zip(u.T,uu.T)]; ids=np.arange(len(u));
for _ in range(5):
 co=np.array([ls(f[ids],v[ids,j]) for j,f in enumerate(a)]); err=np.linalg.norm(np.column_stack([f@c for f,c in zip(a,co)])-v,axis=1); ids=np.argsort(err)[:45]
print('A',co,np.quantile(err,[0,.5,.9,.93,.96,1])); u,v=s['panel_b'].values()
def fp(q,p):return aff(q+p[1]*np.sin(p[0]*q))
def er(p):
 a=fp(u,p); return (a@ls(a,v)-v).ravel()
p=least_squares(er,[2.85,-.4],bounds=([1.5,-1.],[4.5,-.05])).x; print('B',p,np.std(er(p)),ls(fp(u,p),v))
for name,b in [('raw',x),('cal',z),('B',w),('cal+B',np.c_[z,w])]:
 for deg in (1,2,3):
  if deg==3 and b.shape[1]>4: continue
  a=np.c_[np.ones(len(b)),*[np.prod(b[:,i],axis=1) for d in range(1,deg+1) for i in combinations_with_replacement(range(b.shape[1]),d)]]; beta=ls(a[:52],np.c_[y,tr][:52]); r=a@beta-np.c_[y,tr]; print(name,deg,a.shape[1],np.abs(r[:52,0]).mean(),np.abs(r[52:,0]).mean(),np.abs(r[52:,1:]).mean())
a=m['_scalar_features'](w); b=ls(a[:52],y[:52]); r=y-a@b; print('old scalar beta',b,'err portions',[(np.abs(r[i:i+10]).mean(),np.max(np.abs(r[i:i+10]))) for i in range(0,72,10)]); print('worst rows [x z w r]',np.c_[x,z,w,r][np.argsort(abs(r))[-10:]])
