import importlib.util,pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; X,Y=sw['ref_rosette'],sw['raw_rosette']; bank=sw['single_bank_sweeps']; gross=np.max(abs(Y),axis=1)>8
rows=[]
for j in range(4):
 mask=bank['bank_id']==j; xb=np.zeros((mask.sum(),4)); xb[:,j]=bank['ref_signal'][mask]; rows.append((np.r_[X,xb],np.r_[Y[:,j],bank['raw_readout'][mask]]))
def predict(p,x,j,kind):
 a,b,c=x[:,j],x[:,(j+1)%4],x[:,(j-1)%4]; gain=p[4+j]; r=p[8:12]; nonlinear=(a+r[3]*a**3)/(1+p[12]*a*a) if kind=='pade' else a+r[3]*a**3
 return p[j]+gain*(nonlinear+r[0]*b+r[1]*c+r[2]*a*b)
errors={k:[] for k in ('saved','shared','pade')}
for fold in range(8):
 test=np.arange(len(X))%8==fold; train=~test; f=m._calibration(dict(sw,ref_rosette=X[train],raw_rosette=Y[train])); valid=test&~gross
 errors['saved'].append(np.column_stack([m._design(X[valid],j)@f[j]-Y[valid,j] for j in range(4)]))
 for kind in ('shared','pade'):
  init=np.r_[f[:,0],f[:,1],np.mean(f[:,2:]/f[:,1,None],axis=0)];
  if kind=='pade': init=np.r_[init,0.]; bounds=(np.r_[np.full(12,-np.inf),-.08],np.full(13,np.inf))
  else: bounds=(-np.inf,np.inf)
  def residual(p):
   return np.concatenate([(predict(p,x,j,kind)-y)[np.r_[train&~gross,np.ones(len(x)-len(X),bool)]] for j,(x,y) in enumerate(rows)])
  p=least_squares(residual,init,bounds=bounds).x
  errors[kind].append(np.column_stack([predict(p,X[valid],j,kind)-Y[valid,j] for j in range(4)]))
for kind,e in errors.items():
 print(kind,'pooled',np.sqrt(np.mean(np.concatenate(e)**2)),'worst fold x channel',max(np.max(np.sqrt(np.mean(z*z,axis=0))) for z in e))
# Distinguish a nuisance-driven wall comparison from differences in projected source response.
from scipy.linalg import lstsq
S,U=m._forward(s); y=m._calibrate(s); Z=s['Z_nuisance']; idx=np.arange(len(y))//len(s['sensor_ij']); A,b=m._project(s,S,y); o=m.solve_batch(s); w=o['weights']; c=o['kkt_certificate']; d=np.array(c['lam_lower']); u=np.array(c['lam_upper']); cap=s['u_box']; g=A.T@(A@w-b)+s['source_ridge']*w
print('certificate stationarity',max(abs(g-c['nu_sum']-d+u)),'complementarity',max(max(abs(d*w)),max(abs(u*(cap-w)))),'terminal solve residual',np.max(abs((m._operator(s)@o['terminal_field'].ravel(order='F')+s['mu_shifts'][-1]*o['terminal_field'].ravel(order='F'))-np.tensordot(w,s['source_modes'],axes=1).ravel(order='F'))))
for k in np.unique(idx):
 t=idx!=k; v=~t; r=y-S@w; gamma=lstsq(Z[t],r[t])[0]; print('shift',k,'nuisance profile held-out residual RMSE',np.sqrt(np.mean((r[v]-Z[v]@gamma)**2)))
