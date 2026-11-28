import sys,pickle,json,itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
st=calibrate_upstream({**item['study'],'reference_tracer_pulses':item['campaign_diagnostics']}); pulses=st['reference_tracer_pulses']; print('pulses',len(pulses),'keys',list(pulses[0]),flush=True)
W=np.stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); print('W',W.shape,'rank',np.linalg.matrix_rank(W),'Y range',Y.min(),Y.max()); print('aux means',st['aux_manifold_telemetry'].mean(axis=0));
for i,v in enumerate(json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))):
 if isinstance(v,dict) and 'facility_telemetry_trace' in v: print('log',i,v['completed_campaign_short_name'],np.round(np.mean(v['facility_telemetry_trace'],0),3))
na=[st['nx'],st['ny']]; h=[st['Lx']/(na[0]-1),st['Ly']/(na[1]-1)]; ax=[]; opts=[]; ix,iy=st['sensor_ij'].T; at=ix+na[0]*iy; N=np.prod(na); I=sparse.eye(N,format='csc'); V=sparse.diags(st['V_reaction_2d'].ravel(order='F')); rhs=(np.einsum('pm,mij->pij',W,st['source_modes'])).transpose(0,2,1).reshape(len(W),N).T
for d,k in enumerate('xy'):
 kap=st['kappa_'+k]; vel=st['vel_'+k]; n=na[d]; row=np.arange(n); prev=(row-1)%n; nxt=(row+1)%n; left=(kap+kap[prev])/(2*h[d]**2)+np.maximum(vel,0)/h[d]; right=(kap+kap[nxt])/(2*h[d]**2)+np.maximum(-vel,0)/h[d]; A=np.zeros((n,n)); A[row,row]=left+right; A[row,prev]=-left; A[row,nxt]=-right; ends=[0,n-1]; nbr=[1,n-2]; cross=[n-1,0]; ax.append(A); robin=st['robin_'+k]/h[d]; f=kap[ends]/h[d]**2; p=np.array([[-right[0],-left[0],0],[-left[-1],-right[-1],0]]); opts.append({'open':np.column_stack((-f,np.zeros(2),[robin]*2)),'sealed':np.column_stack((-f,np.zeros(2),np.zeros(2))),'periodic':p,'unit':np.column_stack((np.zeros(2),np.zeros(2),np.ones(2))),'pin':np.column_stack((np.zeros(2),np.zeros(2),f))});
def pred(q):
 mats=[]
 for A,bc in zip(ax,q.reshape(2,2,3)):
  A=A.copy(); n=len(A); e=[0,n-1]; A[e,:]=0; A[e,[1,n-2]]=bc[:,0]; A[e,[n-1,0]]=bc[:,1]; A[e,e]=bc[:,2]-bc[:,0]-bc[:,1]; mats.append(sparse.csc_matrix(A))
 L=sparse.kron(sparse.eye(na[1]),mats[0],format='csc')+sparse.kron(mats[1],sparse.eye(na[0]),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[at] for mu in st['mu_shifts']])
models=[]
for ks in itertools.product(*[list(o) for o in opts]):
 q=np.array([o[k] for o,k in zip(opts,ks)]).ravel(); r=np.linalg.norm(pred(q)-Y,axis=0)/np.linalg.norm(Y,axis=0); models.append((np.mean(r),ks,q,r.max()))
models.sort(key=lambda z:z[0]); print('model ranking',[(k,r,w) for r,k,q,w in models[:10]],flush=True); mask=np.arange(len(W))%4!=0
for _,ks,q,_ in models[:3]:
 for bounded in (True,False):
  lo=np.tile([-np.inf,-np.inf,0],4); hi=np.tile([0,0,np.inf],4); bounds=(lo,hi) if bounded else (-np.inf,np.inf)
  try:
   f=least_squares(lambda t:(pred(t)-Y)[:,mask].ravel(),q,bounds=bounds,x_scale='jac',max_nfev=150,gtol=1e-13,ftol=1e-13,xtol=1e-13); r=np.linalg.norm(pred(f.x)-Y,axis=0)/np.linalg.norm(Y,axis=0); print('fit',ks,bounded,'train max',r[mask].max(),'test max',r[~mask].max(),'parameters',f.x.reshape(4,3),flush=True)
  except Exception as e: print(e)
print('axis candidates',opts)
