import importlib.util, pickle, numpy as np
from scipy.sparse import eye, kron, diags, csc_matrix
from scipy.optimize import least_squares
from itertools import product
p='/tmp/hegel_continual_archive/dev_beta_m100_02_s00_solver.py'
a=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(a); a.loader.exec_module(m)
q=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=q['study']; pulses=q['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]).T
Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); Z=s['Z_nuisance']; Q=np.eye(len(Z))-Z@np.linalg.pinv(Z)
print('pulse scale',Y.min(),Y.max(),'Z correlations',np.linalg.norm(Y-Q@Y,axis=0))
def wall(axis, pair):
 k=s['kappa_'+axis]; h=s['L'+axis]/(len(k)-1); r=s['robin_'+axis]; v=s['vel_'+axis]; A=m._axis(k,v,h,r).tolil()
 for i,j,opp,par in [(0,1,len(k)-1,pair[0]),(len(k)-1,len(k)-2,0,pair[1])]:
  fac,sink,wrap=par; D=k[i]/h**2; A[i,j]=-D*fac; A[i,opp]=-D*wrap; A[i,i]=D*(fac+wrap)+r/h*sink
 return A.tocsc()
def op(p):
 nx,ny=s['nx'],s['ny']; ax,ay=wall('x',p[:2]),wall('y',p[2:]); return kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def err(p): return m._responses(s,op(p))[0]@W-Y
kinds={'R':(1,1,0),'N':(1,0,0),'D':(0,1e6,0),'H':(2,1,0),'F':(2,2,0),'P':(1,0,1),'S':(0,1,0)}
print('baseline RMSE by pulse',np.sqrt(np.mean(err([kinds['R']]*4)**2,axis=0)))
rank=[]
for labels in product(kinds,repeat=4):
 e=err([kinds[k] for k in labels]); rank.append((np.linalg.norm(e),''.join(labels),np.linalg.norm(Q@e)))
print('best discrete',sorted(rank)[:12]); print('best projected',sorted(rank,key=lambda t:t[2])[:5]);
for mode in ('symmetric','separate'):
 def params(x):
  a=np.c_[x.reshape(-1,2),np.zeros(len(x)//2)]; return np.repeat(a,2,axis=0) if mode=='symmetric' else a
 for hold in (-1,0,1,2):
  train=[i for i in range(3) if i!=hold]; f=least_squares(lambda x:err(params(x))[:,train].ravel(),np.ones(4 if mode=='symmetric' else 8),bounds=(0,100),gtol=1e-11,xtol=1e-11,ftol=1e-11); e=err(params(f.x)); print(mode,hold,f.x.round(7),'RMSE',np.sqrt(np.mean(e*e,axis=0)),'projected',np.sqrt(np.mean((Q@e)**2,axis=0)))
