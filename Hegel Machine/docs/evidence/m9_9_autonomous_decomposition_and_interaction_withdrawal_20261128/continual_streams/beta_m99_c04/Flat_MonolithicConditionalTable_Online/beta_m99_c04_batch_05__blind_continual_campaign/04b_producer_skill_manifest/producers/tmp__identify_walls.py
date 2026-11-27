import itertools, pickle, runpy
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; pulses=p['campaign_diagnostics']
a=runpy.run_path('/tmp/hegel_continual_archive/beta_m99_c04_batch_04_solver.py')
b=runpy.run_path('/tmp/hegel_continual_archive/beta_m99_c04_batch_02_solver.py')
y=np.stack([a['_calibrate'](dict(s,y_obs_raw=q['y_pulse_raw'])) for q in pulses]); w=np.stack([q['w_pulse'] for q in pulses]); nx,ny=s['nx'],s['ny']; ii,jj=s['sensor_ij'].T
rhs=np.stack([v.ravel(order='F') for v in s['source_modes']],axis=1)@w.T
V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc')
def axis(d,setting):
 k,v=s['kappa_'+d],s['vel_'+d]; h=s['L'+d]/(len(k)-1); r=s['robin_'+d]
 if setting=='periodic': return b['_axis'](k,v,h)
 return a['_axis'](k,v,h,r if setting=='robin' else 0.)
def predict(ax,ay):
 L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.concatenate([splu(L+mu*I).solve(rhs)[ii+nx*jj].T for mu in s['mu_shifts']],axis=1)
def metric(e): return np.sqrt(np.mean(e**2,axis=1)),np.sqrt(np.mean(e.reshape(3,3,-1)**2,axis=2)).max(),abs(e).max()
rank=[]
for x,z in itertools.product(('sealed','periodic','robin'),repeat=2):
 e=predict(axis('x',x),axis('y',z))-y; rank.append((np.linalg.norm(e),x,z)); print(x,z,metric(e))
_,x,z=min(rank); e=(predict(axis('x',x),axis('y',z))-y).reshape(3,3,32)
for j in range(3):
 inds=np.argsort(abs(e[j,0]))[-8:];print('best model pulse',j,[(tuple(s['sensor_ij'][i]),e[j,:,i].round(5).tolist()) for i in inds])
# A diagnostic extension only: each boundary row has an inward coupling,
# endpoint wrap coupling, and net leakage. Unchanged interior rows.
def flexible(d,t):
 A=axis(d,'sealed').tolil();n=A.shape[0];k=s['kappa_'+d];h=s['L'+d]/(n-1)
 for (row,inside,opposite), (inward,wrap,leak) in zip([(0,1,n-1),(n-1,n-2,0)],np.reshape(t,(2,3))):
  A[row,inside]=-inward*k[row]/h**2;A[row,opposite]=-wrap*(k[0]+k[-1])/(2*h**2);A[row,row]=-A[row,inside]-A[row,opposite]+leak/h
 return A.tocsc()
def err(t):return predict(flexible('x',t[:6]),flexible('y',t[6:]))-y
for hold in (None,0,1,2):
 take=[j for j in range(3) if j!=hold]; fit=least_squares(lambda t:err(t)[take].ravel(),np.tile([1.,.2,.1],4),bounds=(np.tile([0.,0.,-.4],4),np.tile([4.,4.,4.],4)),gtol=1e-12,ftol=1e-12,xtol=1e-12);print('flex holdout',hold,'wall rows',fit.x.reshape(4,3).round(5).tolist(),'metrics',metric(err(fit.x)))
