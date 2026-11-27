import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
p=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
d=s['sensor_calibration_sweeps']; x,y=d['ref_rosette'],d['raw_rosette']; A=np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]
Y=np.column_stack([np.linalg.solve(A[1:].T,(t['y_pulse_raw'].reshape(-1,4)-A[0]).T).T.ravel() for t in i['campaign_diagnostics']]); K=np.column_stack([t['w_pulse'] for t in i['campaign_diagnostics']]); B=np.stack([f.ravel(order='F') for f in s['source_modes']],axis=1)@K

def axis(a,scale,leak):
 k,v,n=s['kappa_'+a],s['vel_'+a],s['n'+a]; h=s['L'+a]/(n-1); r=s['robin_'+a]; op=m._axis(k,v,h,r).tolil()
 for j,edge in enumerate([0,n-1]):
  inward=1 if j==0 else n-2; g=scale[j]*k[edge]/h**2
  op[edge,edge]=g+leak[j]*r/h; op[edge,inward]=-g
 return op.tocsc()

def pred(t):
 ax=axis('x',t[:2],t[4:6]); ay=axis('y',t[2:4],t[6:8]); L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(B)[rows] for mu in s['mu_shifts']])

def rmse(e): return np.round(np.sqrt(np.mean(e.reshape(3,len(rows),2)**2,axis=1)),8)
for label,t in [('Robin',np.ones(8)),('reflect',np.r_[np.ones(4),np.zeros(4)]),('reflect2',np.r_[np.full(4,2.),np.zeros(4)]),('reflect_half',np.r_[np.full(4,.5),np.zeros(4)])]:
 e=pred(t)-Y; print(label,rmse(e),'nuisance residual',rmse(e-Q@(Q.T@e)))
for label,expand,start in [('leak',lambda t:np.r_[np.ones(4),t],np.full(4,.1)),('shared',lambda t:np.r_[np.repeat(t[:2],2),np.repeat(t[2:],2)],np.r_[1.,1.,.1,.1]),('full',lambda t:t,np.r_[np.ones(4),np.full(4,.1)])]:
 for held in [-1,0,1,2]:
  use=np.repeat(np.arange(3)!=held,len(rows)); fit=least_squares(lambda t:(pred(expand(t))-Y)[use].ravel(),start,bounds=(0,8),gtol=1e-12,ftol=1e-12,xtol=1e-12); print(label,'held',held,'params',expand(fit.x).round(6),'RMSE',rmse(pred(expand(fit.x))-Y))
print('reaction range',s['V_reaction_2d'].min(),s['V_reaction_2d'].max()); print('cal single-bank max errors',np.max(np.abs(y-(A[0]+x*A[1:].diagonal())),axis=0))
