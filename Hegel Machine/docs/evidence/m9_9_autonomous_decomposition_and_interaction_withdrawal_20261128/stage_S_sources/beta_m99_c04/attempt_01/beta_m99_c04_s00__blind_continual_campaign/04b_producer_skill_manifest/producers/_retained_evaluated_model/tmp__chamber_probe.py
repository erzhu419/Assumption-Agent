import json,pickle
from pathlib import Path
import numpy as np
from scipy.sparse import diags,kron,eye,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
log=json.loads(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').read_text())
for item in log:
 a=np.array(item['facility_telemetry_trace']); print('archive',item['completed_campaign_short_name'], a.shape);print(a)
print('aux',np.round(s['aux_manifold_telemetry'][:6],4))
C=s['sensor_calibration_sweeps']; x=C['ref_rosette']; raw=C['raw_rosette']; print('obs sd',np.std(s['y_obs_raw'])); print('bank',s['row_bank_id'][:36]); print('sensor limits',s['sensor_ij'].min(0),s['sensor_ij'].max(0));
idx=np.arange(len(x)); groups=np.array_split(idx,5)
for degree in [0,1,2,3]:
 if degree==0:
  print('identity calibration rms',np.sqrt(np.mean((raw-x)**2)));continue
 xx=np.column_stack([np.ones(len(x))]+[x**i for i in range(1,degree+1)])
 e=[]
 for fold in groups:
  tr=np.setdiff1d(idx,fold); e.append(np.sqrt(np.mean((raw[fold]-xx[fold]@np.linalg.lstsq(xx[tr],raw[tr],rcond=None)[0])**2)))
 print('degree',degree,'fold rms',e)
M=np.linalg.lstsq(np.c_[x,np.ones(len(x))],raw,rcond=None)[0]; print('calibration',M)
y=np.linalg.solve(M[:4].T,(s['y_obs_raw'].reshape(-1,4)-M[4]).T).T.ravel(); Z=s['Z_nuisance']; fitz=lambda a: np.linalg.lstsq(Z,a,rcond=None)[0]; resid=lambda a: a-Z@fitz(a)
print('nuisance coefficient',fitz(y),'projected RMS',np.sqrt(np.mean(resid(y)**2)))
nx,ny=s['nx'],s['ny']; sensor=s['sensor_ij']; f=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); V=diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc')
def axis(q,ends):
 k=s['kappa_'+q]; v=s['vel_'+q]; n=len(k); h=s['L'+q]/(n-1); rob=s['robin_'+q]; a=np.zeros((n,n))
 for i in range(1,n-1):
  a[i,i-1]=-(k[i-1]+k[i])/(2*h*h)-max(v[i],0)/h
  a[i,i+1]=-(k[i+1]+k[i])/(2*h*h)-max(-v[i],0)/h
  a[i,i]=-a[i,i-1]-a[i,i+1]
 for j,t,other,mode in [(0,1,n-1,ends[0]),(n-1,n-2,0,ends[1])]:
  if mode=='periodic':
   a[j,t]=-(k[j]+k[t])/(2*h*h)-max(-np.sign(t-j)*v[j],0)/h; a[j,other]=-(k[j]+k[other])/(2*h*h)-max(np.sign(t-j)*v[j],0)/h; a[j,j]=-a[j,t]-a[j,other];continue
  kap=k[j]; rr=rob; adv=0
  if mode=='closed': rr=0
  if mode=='strong': rr*=2
  if mode=='dirichlet': rr=1e5
  if mode=='half': rr/=2
  if mode=='fvm': kap=k[j]+k[t]; rr*=2
  if mode=='avg': kap=(k[j]+k[t])/2
  if mode=='adv': adv=max(-np.sign(t-j)*v[j],0)/h
  if isinstance(mode,(float,int)): rr*=mode
  a[j,t]=-kap/h**2-adv; a[j,j]=-a[j,t]+rr/h
 return csc_matrix(a)
def design(ends):
 A=kron(eye(ny),axis('x',ends[:2]))+kron(axis('y',ends[2:]),eye(nx))+V
 return np.concatenate([splu(csc_matrix(A+mu*I)).solve(f)[sensor[:,0]+nx*sensor[:,1]] for mu in s['mu_shifts']])
def fit(D,sel=np.arange(len(y)),reg=1e-7):
 N=Z[sel]; R=lambda a: a-N@np.linalg.lstsq(N,a,rcond=None)[0]; T=R(D[sel]); z=R(y[sel]); m=D.shape[1]
 H=T.T@T+reg*np.eye(m); b=T.T@z
 sol=minimize(lambda w: .5*w@H@w-b@w,np.full(m,s['total_budget']/m),jac=lambda w:H@w-b,method='SLSQP',bounds=[(0,s['u_box'])]*m,constraints={'type':'eq','fun':lambda w:w.sum()-s['total_budget'],'jac':lambda w:np.ones(m)},options={'ftol':1e-13})
 return sol.x,np.linalg.lstsq(N,y[sel]-D[sel]@sol.x,rcond=None)[0]
if __name__=='__main__':
 modes=['open','closed','half','strong','fvm','avg','adv','periodic','dirichlet']; options=set(); options.add(('open',)*4)
 for mode in modes[1:]:
  for mask in range(1,16):options.add(tuple(mode if (mask>>i)&1 else 'open' for i in range(4)))
 scores=[]
 for end in options:
  D=design(end); w,g=fit(D); train=np.sqrt(np.mean((y-D@w-Z@g)**2)); cv=[]
  for test in np.array_split(np.arange(len(y)),3):
   w1,g1=fit(D,np.setdiff1d(np.arange(len(y)),test)); cv.append(np.sqrt(np.mean((y[test]-D[test]@w1-Z[test]@g1)**2)))
  scores.append((train,max(cv),end,np.round(w,4)))
 scores.sort(key=lambda a:a[0]); print('baseline',next(z for z in scores if z[2]==('open',)*4)); print('best'); print(*scores[:12],sep='\n')
