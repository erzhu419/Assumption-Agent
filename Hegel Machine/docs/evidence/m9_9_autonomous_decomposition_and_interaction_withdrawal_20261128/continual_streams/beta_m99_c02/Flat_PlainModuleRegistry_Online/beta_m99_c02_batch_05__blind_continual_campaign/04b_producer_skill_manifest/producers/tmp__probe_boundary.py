import sys,pickle,numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
pulses=item['campaign_diagnostics']; st=calibrate_upstream(dict(item['study'],reference_tracer_pulses=pulses)); pulses=st['reference_tracer_pulses']; w=np.array([p['w_pulse'] for p in pulses]).T; y=np.array([p['y_pulse_raw'] for p in pulses]).T
nx,ny=st['nx'],st['ny']; ij=st['sensor_ij']; rhs=np.column_stack([m.ravel(order='F') for m in st['source_modes']]); shape=(len(st['mu_shifts']),len(ij),len(pulses)); Z=st['Z_nuisance']; q=np.linalg.qr(Z)[0]; print('pulse ranges',y.min(),y.max()); print('aux mean',np.mean(st['aux_manifold_telemetry'],axis=0)); print('sensor ordering',ij[:8].tolist(),st['row_bank_id'][:12].tolist())
def axis(a,p):
 k,v=st['kappa_'+a],st['vel_'+a]; n=len(k); h=st['L'+a]/(n-1); f=(k[:-1]+k[1:])/(2*h*h); wrap=(k[0]+k[-1])/(2*h*h); lo=-f-np.maximum(v[1:],0)/h; hi=-f-np.maximum(-v[:-1],0)/h; main=np.r_[0,-lo[:-1]-hi[1:],0]; A=diags((lo,main,hi),(-1,0,1)).tolil(); r=st['robin_'+a]/h
 for j,i in enumerate((0,n-1)):
  local=k[i]/h**2; near=i+1 if j==0 else i-1; adv=hi[0] if j==0 else lo[-1]; cross=-wrap-max(v[0] if j==0 else -v[-1],0)/h
  leak,blend,join=p[j*3:j*3+3]; A[i,near]=-local+blend*(adv+local); A[i,n-1-i]=join*cross; A[i,i]=-A[i,near]-A[i,n-1-i]+r*leak
 return A.tocsc()
def response(p, modes=rhs):
 L=kron(eye(ny),axis('x',p[:6]))+kron(axis('y',p[6:]),eye(nx))+diags(st['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc'); return np.vstack([splu((L+mu*I).tocsc()).solve(modes)[ij[:,0]+nx*ij[:,1]] for mu in st['mu_shifts']])
def report(label,r): print(label,'rmse',np.sqrt(np.mean(r*r)), 'per shift/pulse',np.sqrt(np.mean(r.reshape(shape)**2,axis=1)).round(7).tolist(),'projected',np.sqrt(np.mean((r-q@(q.T@r))**2)))
models={'open':np.tile([1,0,0],4),'reflect':np.zeros(12),'periodic':np.tile([0,1,1],4)}
for label,p in models.items(): report(label,response(p,rhs@w)-y)
for mask in (np.ones(len(pulses),bool),*(np.arange(len(pulses))!=j for j in range(len(pulses)))):
 def resid(p): return (response(p,rhs@w[:,mask])-y[:,mask]).ravel()
 fit=least_squares(resid,models['open'].astype(float),bounds=(-2,4),max_nfev=120); print('fit',mask.tolist(),np.round(fit.x,5).tolist()); report('fit',response(fit.x,rhs@w)-y)
