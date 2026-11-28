import pickle,runpy,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix,kron,eye,diags
m=runpy.run_path('/tmp/hegel_continual_archive/dev_beta_m100_02_s00_solver.py'); d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=d['study']; X=np.array([p['w_pulse'] for p in d['campaign_diagnostics']]).T; Y=np.array([m['_calibrate'](dict(s,y_obs_raw=p['y_pulse_raw'])) for p in d['campaign_diagnostics']]).T
axes=[]; leak=[]
for q in 'xy':
 h=s['L'+q]/(s['n'+q]-1); a=m['_axis'](s['kappa_'+q],s['vel_'+q],h,s['robin_'+q]).toarray(); axes.append(a); leak.extend([s['robin_'+q]/h]*2)
nx,ny=s['nx'],s['ny']; R=diags(s['V_reaction_2d'].ravel(order='F')); ix,iy=eye(nx),eye(ny)
def predict(p,typ):
 mats=[]
 for j,a in enumerate(axes):
  b=a.copy()
  for k,(i,n,o) in enumerate([(0,1,-1),(-1,-2,0)]):
   t=2*j+k; scale=1 if typ==0 else p[4+t]; wrap=0 if typ<2 else p[8+t]; b[i,n]=a[i,n]*scale; b[i,o]=wrap; b[i,i]=p[t]*leak[t]-b[i,n]-wrap
  mats.append(csc_matrix(b))
 L=kron(iy,mats[0],format='csc')+kron(mats[1],ix,format='csc')+R
 return m['_responses'](s,L)[0]@X
if __name__=='__main__':
 for t in range(3):
  p0=np.r_[np.full(4,.15),np.ones(4*t)]; p0[8:]=0
  for inds in [list(range(3)),[0,1],[0,2],[1,2]]:
   f=least_squares(lambda p:(predict(p,t)-Y)[:,inds].ravel(),p0,ftol=1e-12,xtol=1e-12,gtol=1e-11); e=(predict(f.x,t)-Y).reshape(3,-1,3); print(t,inds,'params',f.x.round(7),'RMSE shift/pulse',np.sqrt(np.mean(e**2,axis=1)).round(8))
 for p in [np.zeros(4),np.ones(4)]:
  print('explicit leak',p[0],np.sqrt(np.mean((predict(p,0)-Y).reshape(3,-1,3)**2,axis=1)))
 print('axis endpoint diffusivities',[-a[i,n] for a in axes for i,n in [(0,1),(-1,-2)]], 'baseline leaks',leak)
