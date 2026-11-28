import numpy as np,pickle,runpy
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52
np.set_printoptions(precision=4,suppress=True,linewidth=160)
m=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a02_batch_06_solver.py'); c=m['_calibration'](s['panel_a']); print('cal',c); u=s['panel_a']['u']; v=s['panel_a']['v']; bi,sl,cr,bl,cu=c; print('cal residual',np.sort(np.linalg.norm(bi+sl*u+(cr+bl*u)*np.roll(u,-1,axis=1)+cu*u**3-v,axis=1)))
z=m['_inverse'](tr['x'],c); print('cal u range',z.min(axis=0),z.max(axis=0))
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
def p2(x):
 i,j=np.triu_indices(x.shape[1]);return np.c_[np.ones(len(x)),x,x[:,i]*x[:,j]]
def score(d,y):return np.mean(abs(d[n:]@ls(d[:n],y[:n])-y[n:]))
for p in ('panel_b','panel_c'):
 u=s[p]['u'];
 for k,v in s[p].items():
  if k=='u':continue
  print(p,k,'quadratic weights\n',ls(p2(u),v));
  if k=='v2':print('residual rms',np.sqrt(np.mean((p2(u)@ls(p2(u),v)-v)**2)))
  if k=='v':
   jj=np.arange(4); vv=np.roll(u,-1,axis=1); ww=np.roll(u,-2,axis=1)
   for j in jj:
    out=v[:,j]; print('B channel',j,'corr',np.corrcoef(np.c_[u,u*u].T,out)[-1,:-1]); print(np.c_[u,out][:10]);
a,q,b=np.linalg.svd(tr['traj'][:n],full_matrices=False); h=tr['traj']@b[0]; k=tr['traj']@b[1]; print('trajectory leading',b[:2], 'ampl range',h.min(),h.max(),'y to trajectory',np.corrcoef(tr['y'],h)[0,1]);
for name,x in [('raw',tr['x']),('calibrated',z)]:
 print(name,'affine',score(p2(x)[:,:5],tr['y']),score(p2(x)[:,:5],h),'quadratic',score(p2(x),tr['y']),score(p2(x),h)); print('dynamics affine amplitude residual',score(np.c_[np.ones(len(h)),h],tr['y']),np.corrcoef(h,k)[0,1]);
