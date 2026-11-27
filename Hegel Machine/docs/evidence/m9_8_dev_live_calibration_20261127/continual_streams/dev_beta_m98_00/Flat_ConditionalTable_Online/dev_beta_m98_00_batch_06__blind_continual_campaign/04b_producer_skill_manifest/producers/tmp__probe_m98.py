import sys, pickle, itertools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m98_00_s00_solver as robin
import dev_beta_m98_00_batch_01_solver as sealed
import dev_beta_m98_00_batch_03_solver as periodic
import dev_beta_m98_00_batch_04_solver as nonlinear
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; sweeps=s['sensor_calibration_sweeps']; ref=sweeps['ref_rosette']; raw=sweeps['raw_rosette']; pulses=it['campaign_diagnostics']; z=s['Z_nuisance']; q=np.linalg.qr(z)[0]
print('fit implementation:'); import inspect; print(inspect.getsource(nonlinear._fit_rosette))
a=np.column_stack([np.ones(len(ref)),ref]); fits=nonlinear._fit_rosette(ref,raw); b,g,f,p,m,c=fits; nxt=np.roll(ref,-1,axis=1); prev=np.roll(ref,1,axis=1); print('sweep residual affine / curved',np.max(np.abs(a@np.linalg.lstsq(a,raw,rcond=None)[0]-raw)),np.max(np.abs(b+g*ref+f*nxt+p*prev+m*ref*nxt+c*ref**3-raw))); print('curvature',m,c)
y={name:[mod._calibrate(dict(s,y_obs_raw=pu['y_pulse_raw'])) for pu in pulses] for name,mod in [('affine',robin),('curved',nonlinear)]}
nx,ny=s['nx'],s['ny']; rows=s['sensor_ij'] @ [1,nx]; rhs=np.column_stack([v.ravel(order='F') for v in s['source_modes']]); eye=sparse.eye(nx*ny,format='csc')
axes=[]
for axis,n in [('x',nx),('y',ny)]:
    args=s['kappa_'+axis],s['vel_'+axis],s['L'+axis]/(n-1)
    axes.append({'R':robin._axis(*args,s['robin_'+axis]),'S':sealed._axis(*args),'P':periodic._axis(*args)})
for x,yy in itertools.product('RSP',repeat=2):
    op=sparse.kron(sparse.eye(ny),axes[0][x],format='csc')+sparse.kron(axes[1][yy],sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    d=np.vstack([splu(op+mu*eye).solve(rhs)[rows] for mu in s['mu_shifts']]); results=[]
    for name in y:
        residual=np.array([d@pu['w_pulse']-o for pu,o in zip(pulses,y[name])]); rr=residual-(residual@q)@q.T; results.append((name, np.round(np.sqrt(np.mean(residual.reshape(2,3,-1)**2,axis=2)),8).tolist(), float(np.sqrt(np.mean(rr**2)))))
    print(x+yy,results)
