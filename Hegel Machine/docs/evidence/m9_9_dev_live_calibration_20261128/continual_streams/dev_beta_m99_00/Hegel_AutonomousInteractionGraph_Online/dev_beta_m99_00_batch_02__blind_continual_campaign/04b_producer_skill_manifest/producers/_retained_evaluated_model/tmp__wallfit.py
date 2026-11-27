import sys,json,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p=json.load(open('/holdout_task/_upstream_param_payload.json')); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['study']; s=calibrate_upstream(raw,p)
W=np.array([d['w_pulse'] for d in item['campaign_diagnostics']]); Y=np.array([calibrate_upstream({**raw,'y_obs_raw':d['y_pulse_raw']},p)['y_obs_raw'] for d in item['campaign_diagnostics']]).T
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; pos=ij[:,0]+nx*ij[:,1]; I=eye(nx*ny,format='csc'); rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W.T

def axis(k,v,h,r,c):
 f=(k[:-1]+k[1:])/(2*h*h); low=np.r_[-f[:-1]-np.maximum(v[1:-1],0)/h,-c[3]*k[-1]/h**2]; up=np.r_[-c[2]*k[0]/h**2,-f[1:]-np.maximum(-v[1:-1],0)/h]; diag=np.r_[c[0]*r/h-up[0],-low[:-1]-up[1:],c[1]*r/h-low[-1]]
 return diags((low,diag,up),(-1,0,1),format='csc')
def forward(c):
 ax,ay=[axis(s['kappa_'+k],s['vel_'+k],s['L'+k]/(n-1),s['robin_'+k],c[i:i+4]) for k,n,i in [('x',nx,0),('y',ny,4)]]
 L=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.vstack([splu(L+z*I).solve(rhs)[pos] for z in s['mu_shifts']])

def report(name,y):
 e=y-Y; print(name,'RMSE per pulse',np.sqrt(np.mean(e*e,axis=0)),'max error',np.max(abs(e)))
if __name__=='__main__':
 from itertools import product
 candidates=[]
 for x0,x1,y0,y1 in product([0,.25,.5,1,2],repeat=4):
  c=[x0,x1,1,1,y0,y1,1,1]; candidates.append((np.linalg.norm(forward(c)-Y),c))
 for _,c in sorted(candidates)[:6]:report(c,forward(c))
 for held in [None,0,1,2]:
  tr=[i for i in range(3) if i!=held]
  for full in [False,True]:
   expand=(lambda z:z) if full else (lambda z:np.r_[z[:2],1,1,z[2:],1,1])
   z=least_squares(lambda z:(forward(expand(z))[:,tr]-Y[:,tr]).ravel(),np.ones(8 if full else 4),bounds=(0,10),gtol=1e-13,xtol=1e-13,ftol=1e-13)
   print('held',held,'rank',np.linalg.svd(z.jac,compute_uv=False)); report(np.round(expand(z.x),7),forward(expand(z.x)))
 for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
  t=np.array(e['facility_telemetry_trace']); print(e['completed_campaign_short_name'],t.mean(0).round(4),t.std(0).round(4))
