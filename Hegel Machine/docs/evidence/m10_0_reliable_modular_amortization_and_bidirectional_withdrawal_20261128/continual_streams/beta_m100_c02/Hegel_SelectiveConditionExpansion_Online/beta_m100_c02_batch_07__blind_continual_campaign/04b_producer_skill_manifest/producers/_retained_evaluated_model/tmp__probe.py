import sys
sys.path.insert(0,'/holdout_task')
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
import pickle,json
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
def cal(y): return calibrate_upstream(dict(s,y_obs_raw=y),pay)['y_obs_raw']
w=np.array([p['w_pulse'] for p in item['campaign_diagnostics']]).T
y=np.array([cal(p['y_pulse_raw']) for p in item['campaign_diagnostics']]).T
nx,ny=s['nx'],s['ny']; sensors=tuple(s['sensor_ij'].T); inds=sensors[0]+nx*sensors[1]

def ax(tag,n,v):
 h=s['L'+tag]/(n-1); k=s['kappa_'+tag]; vel=s['vel_'+tag]; rob=s['robin_'+tag]; a=np.zeros((n,n))
 km=(k[1:-1]+k[:-2])/2; kp=(k[1:-1]+k[2:])/2; i=np.arange(1,n-1)
 a[i,i-1]=-km/h**2-np.maximum(vel[i],0)/h; a[i,i+1]=-kp/h**2-np.maximum(-vel[i],0)/h; a[i,i]=-a[i,i-1]-a[i,i+1]
 for j,b,d in [(0,1,v[:3]),(-1,-2,v[3:])]:
  a[j,j]=k[j]/h**2+rob/h+d[0]; a[j,b]=-k[j]/h**2+d[1]; a[j,n-1 if j==0 else 0]=d[2]
 return a
rhs=np.stack([m.ravel(order='F') for m in s['source_modes']],axis=1)@w

def pred(v):
 a=sp.kronsum(sp.csc_matrix(ax('x',nx,v[:6])),sp.csc_matrix(ax('y',ny,v[6:])),format='csc')
 return np.concatenate([splu(a+sp.diags(s['V_reaction_2d'].ravel(order='F')+mu,format='csc')).solve(rhs)[inds] for mu in s['mu_shifts']])
if __name__=='__main__':
 z=s['Z_nuisance']; q=np.linalg.qr(z)[0]
 for project in [False,True]:
  def res(v):
   r=pred(v)-y
   return r-q@(q.T@r) if project else r
  def show(r): return np.round(np.sqrt(np.mean(r.reshape(3,32,3)**2,axis=1)),9)
  v=np.zeros(12);print('baseline',project,show(res(v)))
  for free in [[0,3,6,9],[0,1,3,4,6,7,9,10],list(range(12))]:
   def expand(p):
    out=v.copy();out[free]=p;return out
   a=least_squares(lambda p:res(expand(p)).ravel(),v[free],max_nfev=200,gtol=1e-13,ftol=1e-12,xtol=1e-12)
   print('fit',free,np.round(expand(a.x),7),show(res(expand(a.x))), 'sv',np.round(np.linalg.svd(a.jac,compute_uv=False),7))
 for tag,n in [('x',nx),('y',ny)]:
  h=s['L'+tag]/(n-1); k=s['kappa_'+tag];print(tag,'h',h,'d',k[[0,-1]]/h**2,'faces',(k[[0,-1]]+k[[1,-2]])/(2*h**2),'r',s['robin_'+tag]/h,'v',s['vel_'+tag][[0,-1]]/h)
