import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import null_space
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=m._Transport(s); c=m._calibrator(s['sensor_calibration_sweeps']); p=s['reference_tracer_pulses']; rhs=t.modes@np.column_stack([v['w_pulse'] for v in p]); Y=np.column_stack([c(v['y_pulse_raw']) for v in p]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; A=s['aux_manifold_telemetry']; w=m._wall_fit(t,p,c); S,U=t.response(w,t.modes); obs=c(s['y_obs_raw']); E=null_space(np.ones((1,S.shape[1]))); w0=np.full(S.shape[1],s['total_budget']/S.shape[1]); D=np.column_stack([S@E,Z]); r=obs-S@w0-D@np.linalg.lstsq(D,obs-S@w0,rcond=None)[0]; print('shape set',[(k,np.shape(s[k])) for k in ('row_bank_id','sensor_ij','aux_manifold_telemetry','Z_nuisance','mu_shifts')]);print('row bank order',s['row_bank_id'][:36]); print('obs projected residual shift/bank RMS',np.sqrt((r.reshape(len(t.shifts),-1,4)**2).mean(axis=1))); print('aux pooled channel residual correlations',np.corrcoef(A.T,r.reshape(-1,4).T)[:4,4:]);print('pulse residual channel RMS',np.sqrt(((t.response(w,rhs)[0]-Y).reshape(-1,4,2)**2).mean(axis=0)));hx=s['Lx']/(s['nx']-1);hy=s['Ly']/(s['ny']-1); print('pulse walls physical',w*np.repeat([hx,hy],2),'nominal',t.nominal*np.repeat([hx,hy],2)); print('diffusion endpoints',s['kappa_x'][[0,-1]],s['kappa_y'][[0,-1]])
# Distinguish remaining interior scale errors from independent wall mismatch on held-out shifts.
nx,ny=t.shape; V=diags(s['V_reaction_2d'].ravel(order='F')).tocsc(); Dx=kron(eye(ny),m._axis(s['kappa_x'],np.zeros(nx),hx)).tocsc(); Dy=kron(m._axis(s['kappa_y'],np.zeros(ny),hy),eye(nx)).tocsc(); Ax=kron(eye(ny),m._axis(np.zeros(nx),s['vel_x'],hx)).tocsc(); Ay=kron(m._axis(np.zeros(ny),s['vel_y'],hy),eye(nx)).tocsc(); bases=[Dx,Dy,Ax,Ay,V]; nobs=len(t.sensors); groups=np.repeat(np.arange(len(t.shifts)),nobs)
def pred(x, terms):
 a=t.matrix+diags(t.walls@x[:4]);
 for j,z in zip(terms,x[4:]): a=a+z*bases[j]
 return np.concatenate([splu(a+mu*t.identity).solve(rhs)[t.sensors] for mu in t.shifts])
for terms in [[],[0,1],[2,3],[4],[0,1,2,3,4]]:
 start=np.r_[w,np.zeros(len(terms))]; vals=[]
 for sh in range(-1,len(t.shifts)):
  train=groups!=sh; sol=least_squares(lambda x:(pred(x,terms)-Y)[train].ravel(),start,gtol=1e-13); err=pred(sol.x,terms)-Y; vals.append(np.mean(err[~train if sh>=0 else train]**2)**.5)
  if sh==-1: full=sol.x
 print('interior terms',terms,'full+held RMSE',vals,'parameters',full)
