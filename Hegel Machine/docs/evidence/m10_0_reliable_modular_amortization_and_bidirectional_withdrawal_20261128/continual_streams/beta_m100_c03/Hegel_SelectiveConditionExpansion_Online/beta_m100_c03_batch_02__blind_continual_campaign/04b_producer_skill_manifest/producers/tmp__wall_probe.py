import sys,json,pickle,itertools
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
Y=np.stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in item['campaign_diagnostics']],axis=1)
W=np.stack([p['w_pulse'] for p in item['campaign_diagnostics']],axis=1)
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; z=s['Z_nuisance']
N=nx*ny; M=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); idx=ij[:,0]+nx*ij[:,1]
def axis(d,p):
 k=s['kappa_'+d]; v=s['vel_'+d]; n=len(k); h=s['L'+d]/(n-1); face=(k[:-1]+k[1:])/(2*h*h)
 l=-face-np.maximum(v[1:],0)/h; r=-face-np.maximum(-v[:-1],0)/h
 l[-1]=-k[-1]/h**2*p[3]; r[0]=-k[0]/h**2*p[0]
 A=diags([l,np.r_[0,-l[:-1]-r[1:],0],r],[-1,0,1],format='lil')
 for i,j,b in [(0,1,p[:3]),(n-1,n-2,p[3:])]:
  A[i,n-1-i]=-b[2]*k[i]/h**2
  A[i,i]=-A[i,j]-A[i,n-1-i]+b[1]*s['robin_'+d]/h
 return A.tocsc()
def calc(p):
 a,b=axis('x',p[:6]),axis('y',p[6:]); I=eye(N,format='csc'); L=(kron(eye(ny),a)+kron(b,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.concatenate([splu(L+mu*I).solve(M@W)[idx] for mu in s['mu_shifts']],axis=0)
def rms(x): return np.sqrt(np.mean(x*x,axis=0))
def prof(x): return x-z@np.linalg.lstsq(z,x,rcond=None)[0]
opts={'open':[1,1,0],'sealed':[1,0,0],'periodic':[1,0,1],'double':[2,0,0],'double_robin':[2,2,0],'ghost_wrap':[0,0,1]}
if __name__=='__main__':
 print('pulse weights',W.T,'sample Y',Y[:4], 'baseline noise along nuisance',rms(Y-prof(Y)))
 scores=[]
 for x,y in itertools.product(opts,repeat=2):
  p=np.array(opts[x]*2+opts[y]*2); e=calc(p)-Y; scores.append((max(rms(prof(e))),x,y,rms(e),rms(prof(e))))
 for row in sorted(scores)[:12]: print(row)
 def sym(p): return np.repeat(p.reshape(2,3),2,axis=0).ravel()
 for grouped in (True,False):
  for train in ([0,1,2],[1,2],[0,2],[0,1]):
   conv=sym if grouped else np.asarray
   def fun(p):return (1e4*prof(calc(conv(p))-Y)[:,train]).ravel()
   fit=least_squares(fun,[1,1,.03]*(2 if grouped else 4), bounds=([.001,0,0]*(2 if grouped else 4),[5,12,5]*(2 if grouped else 4)),max_nfev=200)
   e=calc(conv(fit.x))-Y
   print('symmetric',grouped,'train',train,'params',np.round(fit.x,6),'raw RMS',rms(e),'profile RMS',rms(prof(e)))
