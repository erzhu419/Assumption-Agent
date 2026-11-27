import sys,pickle,numpy as np
from scipy.sparse import eye,kron,diags,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m99_c01_s00_solver as base
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; ds=item['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; n=nx*ny
Y=np.column_stack([base._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds]); W=np.stack([d['w_pulse'] for d in ds],axis=1); rhs=s['source_modes'].reshape(len(W),-1,order='F').T@W
ix,iy=s['sensor_ij'].T; idx=ix+nx*iy
axes=[]
for a,m in [('x',nx),('y',ny)]:
 h=s['L'+a]/(m-1); k=s['kappa_'+a]; v=s['vel_'+a]; r=s['robin_'+a]; axes.append((base._axis(k,v,h,r).toarray(),k[[0,-1]]/h**2,r/h,v[[0,-1]]/h))
I=eye(n,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def predict(p):
 out=[]
 for (A0,k,r,v),p1 in zip(axes,np.reshape(p,(2,2,3))):
  A=A0.copy(); m=len(A)
  for j,(i,near,wrap) in enumerate([(0,1,m-1),(m-1,m-2,0)]):
   f,b,c=p1[j]; A[i,near]=-k[j]*f; A[i,wrap]=-k[j]*c; A[i,i]=k[j]*(f+c)+r*b
  out.append(csc_matrix(A))
 L=(kron(eye(ny),out[0])+kron(out[1],eye(nx))).tocsc()+V
 return np.concatenate([splu(L+mu*I).solve(rhs)[idx] for mu in s['mu_shifts']])
def pmap(q,model):
 q=np.asarray(q)
 if model=='shared': return np.tile([q[0],q[1],0],4)
 if model=='axes': return np.repeat(np.column_stack([q.reshape(2,2),np.zeros(2)]),2,axis=0).ravel()
 if model=='walls': return np.column_stack([q.reshape(4,2),np.zeros(4)]).ravel()
 return q
if __name__=='__main__':
 print('axes k/h², robin/h, vel/h:',[a[1:] for a in axes]); print('calibrated pulse min/max',Y.min(axis=0),Y.max(axis=0),'shifts',s['mu_shifts'])
 for desc,p in [('Robin',[1,1,0]),('reflecting',[1,0,0]),('reflecting half-cell',[2,0,0]),('periodic',[1,0,1])]:
  D=(predict(p*4)-Y).reshape(3,-1,3); print(desc,np.sqrt(np.mean(D**2,axis=1)))
 for model,start in [('shared',[1,0.1]),('axes',[1,0.1]*2),('walls',[1,0.1]*4),('wrap',[1,.1,.01]*4)]:
  print(model)
  for hold in [None,0,1,2]:
   ids=[i for i in range(3) if i!=hold]; f=lambda q:(predict(pmap(q,model))[:,ids]-Y[:,ids]).ravel()
   fit=least_squares(f,start,bounds=(0,np.inf),gtol=1e-11,max_nfev=350); err=np.sqrt(np.mean((predict(pmap(fit.x,model))-Y)**2,axis=0)); print(hold,np.round(fit.x,7),np.round(err,8))
