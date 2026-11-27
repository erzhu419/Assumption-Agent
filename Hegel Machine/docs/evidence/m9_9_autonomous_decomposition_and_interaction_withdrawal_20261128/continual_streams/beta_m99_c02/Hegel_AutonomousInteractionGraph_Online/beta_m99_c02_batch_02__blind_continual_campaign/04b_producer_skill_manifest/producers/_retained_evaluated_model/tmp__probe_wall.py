import sys,json,pickle,itertools
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; p=json.load(open('/holdout_task/_upstream_param_payload.json')); d=it['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; n=nx*ny
inds=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]
M=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); W=np.column_stack([z['w_pulse'] for z in d]); Y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=z['y_pulse_raw']),p)['y_obs_raw'] for z in d]); R=M@W
print('pulse weights',W.T,'target norm',np.linalg.norm(Y),'sensor extrema',s['sensor_ij'].min(0),s['sensor_ij'].max(0))
print('telemetry mean',s['aux_manifold_telemetry'].mean(0))

def axis(a,j):
 k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(len(k)-1); q=len(k); A=np.zeros((q,q)); f=(k[:-1]+k[1:])/(2*h*h)
 for i in range(1,q-1):
  A[i,i-1]=-f[i-1]-max(v[i],0)/h; A[i,i+1]=-f[i]-max(-v[i],0)/h; A[i,i]=-A[i,i-1]-A[i,i+1]
 for t,(i,ii,iii) in enumerate([(0,1,q-1),(q-1,q-2,0)]):
  c=k[i]/h**2; local=f[0 if t==0 else -1]; wrap=(k[0]+k[-1])/(2*h*h); rob=s['robin_'+a]/h
  if not isinstance(j,str):
   mult, loss, cross=j[t]; inward=mult*c; jump=cross*c
  else:
   inward,jump,loss={'open':(c,0,rob),'reflect':(c,0,0),'reflect2':(2*c,0,0),'face':(local,0,0),'face2':(2*local,0,0),'period':(local+max((-1 if t==0 else 1)*v[i],0)/h,wrap+max((1 if t==0 else -1)*v[i],0)/h,0),'dirichlet':(0,0,1e6),'short':(c,0,2*rob),'wrap':(c,wrap,0)}[j]
  A[i,i]=inward+jump+loss; A[i,ii]=-inward; A[i,iii]=-jump
 return A

def pred(x,y):
 L=np.kron(np.eye(ny),axis('x',x))+np.kron(axis('y',y),np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
 return np.concatenate([solve(L+mu*np.eye(n),R)[inds] for mu in s['mu_shifts']])
if __name__=='__main__':
 kinds=['open','reflect','reflect2','face','face2','period','dirichlet','short','wrap']; errors=sorted((np.linalg.norm(pred(x,y)-Y),x,y) for x,y in itertools.product(kinds,repeat=2)); print('wall rankings',errors[:15])
 # Fit all endpoint row entries directly, with one diagonal gauge fixed by centering to the best discrete solution.
 _,x,y=errors[0]; init=[]
 for a,j in [('x',x),('y',y)]:
  A=axis(a,j); k=s['kappa_'+a]; h=s['L'+a]/(len(k)-1)
  for i,ii,iii in [(0,1,len(k)-1),(len(k)-1,len(k)-2,0)]:
   init.extend([-A[i,ii]*h*h/k[i],A[i].sum(),-A[i,iii]*h*h/k[i]])
 def fun(b):
  z=b.reshape(4,3); return (pred(z[:2],z[2:])-Y).ravel()
 b=least_squares(fun,init,x_scale=[1,10,1]*4,gtol=1e-13,ftol=1e-13,xtol=1e-13,max_nfev=100); print('free endpoint RMSE',np.sqrt(np.mean(b.fun**2)),'fit',b.x.reshape(4,3),'sv',np.linalg.svd(b.jac,compute_uv=False)); print('grid',[(a,s['kappa_'+a][[0,-1]],s['vel_'+a][[0,-1]],s['robin_'+a]/(s['L'+a]/(s['n'+a]-1))) for a in 'xy'])
