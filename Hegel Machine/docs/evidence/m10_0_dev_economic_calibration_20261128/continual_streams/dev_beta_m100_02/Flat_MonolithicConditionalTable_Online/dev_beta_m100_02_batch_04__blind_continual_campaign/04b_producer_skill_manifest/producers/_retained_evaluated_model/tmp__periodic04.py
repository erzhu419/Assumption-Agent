import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import eye,kron,diags
p='/tmp/hegel_continual_archive/dev_beta_m100_02_batch_03_solver.py'
a=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(a); a.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; ds=it['campaign_diagnostics']; c=s['sensor_calibration_sweeps']; t=m._sweep_fit(s)
print('coeffs',np.round(t,7)); e=m._raw(c['ref_rosette'],t)-c['raw_rosette']; print('rosette absolute residual quantiles',np.quantile(abs(e),[.5,.9,.95,.98,1],axis=0));
Y=np.stack([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds],axis=1); W=np.stack([d['w_pulse'] for d in ds],axis=1)
nx,ny=s['nx'],s['ny']; Z=s['Z_nuisance']; rz=Z@np.linalg.pinv(Z); hx=s['Lx']/(nx-1); hy=s['Ly']/(ny-1)
ax=m._axis(s['kappa_x'],s['vel_x'],hx,0); ay=m._axis(s['kappa_y'],s['vel_y'],hy,0)
base=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')

def design(p):
    leak,diff=np.asarray(p).reshape(2,4); axes=[]
    for q,(n,h,k,r) in enumerate([(nx,hx,s['kappa_x'],s['robin_x']),(ny,hy,s['kappa_y'],s['robin_y'])]):
        b=np.zeros((n,n)); z=2*q
        for j,i,i2 in [(z,0,1),(z+1,n-1,n-2)]:
            d=diff[j]*k[i]/h**2; b[i,i]=d+leak[j]*r/h; b[i,i2]=-d
        from scipy.sparse import csc_matrix
        axes.append(csc_matrix(b))
    return m._responses(s,base+kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc'))[0]@W

def err(p,project=False):
    e=design(p)-Y
    return e-rz@e if project else e
k=np.asarray(s['kappa_x']); v=np.asarray(s['vel_x']); i=np.array([0,len(k)-1]); j=np.array([1,len(k)-2]); opp=i[::-1]
print('h,k,v',hx,k[[0,1,-2,-1]],v[i]); print('telemetry',s.get('aux_manifold_telemetry',np.empty((0,0))).shape)
for face in ('endpoint','average'):
 for upwind in (False,True):
  d=(0.5*(k[i]+k[j]) if face=='average' else k[i]).copy(); wrap=(0.5*(k[i]+k[opp]) if face=='average' else k[i]).copy()
  if upwind:
   d+=hx*np.maximum([-v[0],v[-1]],0); wrap+=hx*np.maximum([v[0],-v[-1]],0)
  p=np.r_[np.zeros(4),d/k[i]-1,0,0,wrap/k[i],0,0]; print(face,upwind,'params',p[4:],'pulse RMSE',np.sqrt(np.mean(err(p)**2,axis=0)))
# Compare absolute conductances to discover simpler shared face laws if needed.
print('pulse-inferred endpoint conductances * h^2',k[i]*[1.0369,1.0369],k[i]*[1.1422,1.0836]); print('face averages',.5*(k[i]+k[j]),.5*(k[i]+k[opp])); print('upwind correction',hx*v[i])
