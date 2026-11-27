import importlib.util, pickle, json
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/'
spec=importlib.util.spec_from_file_location('old',p+'beta_m98_v2_c01_s00_solver.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; nxs=(s['nx'],s['ny']); n=np.prod(nxs); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; ij=s['sensor_ij']; rows=ij[:,0]+nxs[0]*ij[:,1]; I=eye(n,format='csc'); shifts=s['mu_shifts']; Y=np.array([old._calibrate(dict(s,y_obs_raw=t['y_pulse_raw'])) for t in it['campaign_diagnostics']]).T
rhs=np.stack([np.tensordot(t['w_pulse'],s['source_modes'],axes=1).ravel(order='F') for t in it['campaign_diagnostics']],axis=1)
axes=[]; scale=[]; robin=[]
for ni,tag in zip(nxs,'xy'):
 h=s['L'+tag]/(ni-1); k=s['kappa_'+tag]; v=s['vel_'+tag]; r=s['robin_'+tag]/h
 a=old._axis(k,v,h,s['robin_'+tag]).toarray(); axes.append(a); scale.append(k[[0,-1]]/h**2); robin.append(r)
scale=np.array(scale); robin=np.array(robin)
V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def pred(params):
 ops=[]
 for a,pa,sc,r in zip(axes,params.reshape(2,2,3),scale,robin):
  a=a.copy()
  for i,j,k,b,pp in zip((0,-1),(1,-2),(-1,0),sc,pa):
   loss,near,wrap=pp; a[i,i]=loss*r+(near+wrap)*b; a[i,j]=-near*b; a[i,k]=-wrap*b
  ops.append(diags((np.diag(a,-1),np.diag(a),np.diag(a,1)),(-1,0,1),format='lil')); ops[-1][0,-1]=a[0,-1]; ops[-1][-1,0]=a[-1,0]
 L=kron(eye(nxs[1]),ops[0],format='csc')+kron(ops[1],eye(nxs[0]),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[rows] for mu in shifts])
def err(pa):return pred(pa)-Y
def show(label,pa):
 e=err(pa); print(label,'shift',np.sqrt(np.mean(e.reshape(len(shifts),-1,2)**2,axis=1)).round(8),'prof',np.sqrt(np.mean((e-Q@(Q.T@e))**2)).round(8),flush=True)
choices={'robin':(1,1,0),'reflect':(0,1,0),'reflect2':(0,2,0),'periodic':(0,1,1),'periodic2':(0,.5,.5),'edge_pin':(1,0,0)}
for a in choices:
 for b in choices:
  pa=np.array([choices[a]]*2+[choices[b]]*2)
  e=err(pa); print(a,b,round(np.sqrt(np.mean(e**2)),8),round(np.sqrt(np.mean((e-Q@(Q.T@e))**2)),8))
for projected in [False,True]:
 for init in [(1,1,0),(0,1,0),(0,1,1)]:
  def fun(pa):
   e=err(pa); return (e-Q@(Q.T@e) if projected else e).ravel()
  fit=least_squares(fun,np.tile(init,4).astype(float),max_nfev=150,gtol=1e-12,ftol=1e-12,xtol=1e-12)
  print('fit',projected,init,fit.x.reshape(4,3).round(6),fit.optimality); show('fit metrics',fit.x)
for label,t in [('archive',json.load(open(p+'archive_apparatus_log.json'))[-1]['facility_telemetry_trace']),('current',s['aux_manifold_telemetry'])]:
 t=np.asarray(t);print(label,t.shape,'means',t.mean(0).round(4),'sd',t.std(0).round(4))
print('mean pulses by shift',Y.reshape(len(shifts),-1,2).mean(1))
