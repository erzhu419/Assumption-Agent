import importlib.util,pickle,numpy as np
from scipy import sparse
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/dev_beta_m100_01_s00_solver.py'
spec=importlib.util.spec_from_file_location('old',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; ds=i['campaign_diagnostics']; W=np.array([d['w_pulse'] for d in ds]); Y=np.array([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds]); nx,ny=s['nx'],s['ny']; L=m._operator(s)
terms=[]
for n,k in [(nx,'x'),(ny,'y')]:
 h=s['L'+k]/(n-1); kap=s['kappa_'+k]; rob=s['robin_'+k]
 for edge,j in [(0,1),(n-1,n-2)]:
  for entries in [[(edge,edge,kap[edge]/h**2),(edge,j,-kap[edge]/h**2)],[(edge,edge,rob/h)]]:
   a=sparse.csc_matrix(([v for r,c,v in entries],([r for r,c,v in entries],[c for r,c,v in entries])),shape=(n,n)); terms.append(sparse.kron(sparse.eye(ny),a,format='csc') if k=='x' else sparse.kron(a,sparse.eye(nx),format='csc'))
def res(t):
 A=L+sum(((v-1)*a for v,a in zip(t,terms)),start=0*L)
 S,_=m._forward(s,A)
 return (W@S.T-Y)
def report(label,t):
 r=res(t); print(label,'coeff',np.round(t,7),'rmse',np.sqrt(np.mean(r*r,axis=1)),'max',np.max(np.abs(r)), 'by_shift',np.sqrt(np.mean(r.reshape(3,3,-1)**2,axis=(0,2))))
report('open',np.ones(8)); report('sealed',np.tile([1.,0.],4)); opt=least_squares(lambda t:res(t).ravel(),np.ones(8),bounds=(np.tile([0.01,-1.],4),np.tile([12.,20.],4)),gtol=1e-13,ftol=1e-13,xtol=1e-13); report('general',opt.x); print('singular',np.linalg.svd(opt.jac,compute_uv=False)); print('residual first slice',np.round(res(opt.x)[0,:12],7)); print('boundary kap and vel',[(s['kappa_'+k][[0,-1]],s['vel_'+k][[0,-1]]) for k in 'xy']); print('bank sweep fits', [np.polyfit(s['sensor_calibration_sweeps']['single_bank_sweeps']['ref_signal'][j:j+12],s['sensor_calibration_sweeps']['single_bank_sweeps']['raw_readout'][j:j+12],1) for j in range(0,48,12)])
for j in range(3):
 keep=np.arange(3)!=j; q=least_squares(lambda t:res(t)[keep].ravel(),opt.x,bounds=(np.tile([0.01,-1.],4),np.tile([12.,20.],4)),gtol=1e-12); r=res(q.x)[j]; print('held pulse',j,'RMSE',np.sqrt(np.mean(r*r)),'coeff',np.round(q.x,5))
