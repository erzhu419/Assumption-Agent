import importlib.util,json,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
p='/tmp/hegel_continual_archive/dev_beta_m98_00_batch_02_solver.py'; spec=importlib.util.spec_from_file_location('old',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=s['nx']; k=s['ny']; ij=s['sensor_ij']; idx=ij[:,0]+n*ij[:,1]; shifts=s['mu_shifts']; src=s['source_modes']; cal=m._calibration(s['sensor_calibration_sweeps']); pulses=s['reference_tracer_pulses']; y=np.array([cal(p['y_pulse_raw']) for p in pulses]).T; modes=np.array([np.tensordot(p['w_pulse'],src,axes=1) for p in pulses]); rhs=modes.transpose(0,2,1).reshape(len(modes),-1).T
print('sensors',ij.tolist()); print('shifts',shifts,'pulse range',y.min(),y.max()); print('wall inputs',[(s['kappa_'+a][[0,-1]],s['vel_'+a][[0,-1]]) for a in 'xy'])
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('history means', [np.mean(e['facility_telemetry_trace'],axis=0).round(3) for e in log]); print('aux mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0))
Ax=[]
for a,N in zip('xy',(n,k)):
 h=s['L'+a]/(N-1); kap=s['kappa_'+a]; A=m._axis(kap,s['vel_'+a],h,s['robin_'+a],np.ones(4)).tolil(); D=[]
 for i,j,t in [(0,1,N-1),(N-1,N-2,0)]:
  for z,scale in [(None,s['robin_'+a]/h),(j,kap[i]/h**2),(t,kap[i]/h**2)]:
   E=diags(np.zeros(N)).tolil(); E[i,i]=scale
   if z is not None: E[i,z]=-scale
   D.append(E.tocsc())
 Ax.append((A.tocsc(),D))
B=(kron(eye(k),Ax[0][0])+kron(Ax[1][0],eye(n))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); D=[kron(eye(k),e).tocsc() for e in Ax[0][1]]+[kron(e,eye(n)).tocsc() for e in Ax[1][1]]; I=eye(n*k,format='csc'); x0=np.tile([1.,1.,0.],4)
def pred(x):
 L=B+sum(d*v for d,v in zip(D,x-x0)); o=[]
 for j,mu in enumerate(shifts):
  p=idx if len(idx)*len(shifts)==len(y) else np.array_split(idx,len(shifts))[j]
  o.append(splu(L+mu*I).solve(rhs)[p])
 return np.vstack(o)
for name,inds,lo in [('nonnegative',np.arange(12),0.),('signed_sink',np.arange(12),np.tile([-4.,0.,0.],4)),('signed_sink_no_wrap',np.r_[0,1,3,4,6,7,9,10],np.tile([-4.,0.],4)),('reflecting',np.r_[1,4,7,10],0.)]:
 def expand(x):
  z=x0.copy(); z[inds]=x
  if name=='reflecting':z[::3]=0
  return z
 r=least_squares(lambda x:(pred(expand(x))-y).ravel(),x0[inds],bounds=(lo,30.),max_nfev=140,gtol=1e-12); e=pred(expand(r.x))-y
 print(name,r.nfev,expand(r.x).round(6),np.sqrt(np.mean(e**2,axis=0)),np.sqrt(np.mean(e.reshape(3,-1)**2,axis=1)))
