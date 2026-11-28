import importlib.util,json,pickle
import numpy as np
from scipy.optimize import root, minimize
from scipy import sparse
from scipy.sparse.linalg import splu
base='/tmp/hegel_continual_archive/'
def load(name):
 spec=importlib.util.spec_from_file_location(name,base+'dev_beta_m100_01_'+name+'_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
models={k:load(v) for k,v in [('R','s00'),('N','batch_01'),('P','batch_02')]}
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def design(x,j):
 a,b,c=x[:,j],x[:,(j+1)%4],x[:,(j-1)%4];return np.c_[np.ones(len(x)),a,b,c,a*b,a**3]
def calibration(st, valid=None):
 s=st['sensor_calibration_sweeps'];x=s['ref_rosette'];r=s['raw_rosette']; b=s['single_bank_sweeps']; out=[]; residuals=[]
 if valid is None: valid=np.ones(len(x),bool)
 for j in range(4):
  ii=b['bank_id']==j; xb=np.zeros((sum(ii),4));xb[:,j]=b['ref_signal'][ii]
  A=np.vstack([design(x[valid],j),design(xb,j)]); y=np.r_[r[valid,j],b['raw_readout'][ii]]
  fit=np.linalg.lstsq(A,y,rcond=None)[0]
  for _ in range(6):
   e=y-A@fit; s=max(1e-8,1.4826*np.median(abs(e-np.median(e)))); q=1/np.sqrt(1+(e/(2.5*s))**2);fit=np.linalg.lstsq(A*q[:,None],y*q,rcond=None)[0]
  e=y-A@fit; keep=abs(e)<max(8*1.4826*np.median(abs(e-np.median(e))),2e-4); fit=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0]
  out.append(fit);residuals.append((np.sqrt(np.mean((y[keep]-A[keep]@fit)**2)),np.flatnonzero(~keep)))
 return np.array(out),residuals
def cal(st, c):
 def f(x):return np.array([design(x[None,:],j)[0]@c[j] for j in range(4)])
 w=[]; errs=[]
 for y in st['y_obs_raw'].reshape(-1,4):
  t=root(lambda x:f(x)-y,np.zeros(4),tol=1e-11);w.append(t.x);errs.append(np.max(abs(t.fun)))
 return np.array(w).ravel(),max(errs)
def op(kinds):
 ax=[]
 for v,k in zip('xy',kinds):
  a,vel=st['kappa_'+v],st['vel_'+v];h=st['L'+v]/(len(a)-1)
  ax.append(models[k]._axis(a,vel,h,st['robin_'+v]) if k=='R' else models[k]._axis(a,vel,h))
 nx,ny=st['nx'],st['ny'];return sparse.kron(sparse.eye(ny),ax[0],format='csc')+sparse.kron(ax[1],sparse.eye(nx),format='csc')+sparse.diags(st['V_reaction_2d'].ravel(order='F'),format='csc')
def profile(S,y,keep):
 Z=st['Z_nuisance'];D=np.c_[S,Z];m=S.shape[1];budget=st['total_budget'];cap=st['u_box'];A=D[keep];b=y[keep];x=np.linalg.lstsq(np.r_[A,np.r_[np.ones(m),np.zeros(Z.shape[1])][None,:]*10],np.r_[b,budget*10],rcond=None)[0]
 # Fit the measurement equation, not the final ridge-regularized design objective.
 fun=lambda w: 0.5*np.sum((A@w-b)**2); jac=lambda w:A.T@(A@w-b)
 eq=np.r_[np.ones(m),np.zeros(Z.shape[1])]; t=minimize(fun,x,jac=jac,bounds=[(0,cap)]*m+[(None,None)]*Z.shape[1],constraints=[{'type':'eq','fun':lambda w:eq@w-budget,'jac':lambda w:eq}],method='SLSQP',options={'ftol':1e-14,'maxiter':250})
 return y-D@t.x,t.x
if __name__=='__main__':
 from itertools import product
 fit,diag=calibration(st);y,err=cal(st,fit);print('fit',fit,'trimmed calibration',diag,'inverse error',err,'y range',y.min(),y.max())
 c=st['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; clean=np.max(abs(Y),axis=1)<8; errs=[]
 for i in range(8):
  keep=np.arange(len(X))%8!=i; f,_=calibration(st,keep); val=~keep&clean; errs.append(np.sqrt(np.mean(np.column_stack([design(X[val],j)@f[j]-Y[val,j] for j in range(4)])**2,axis=0)))
 print('CV clean worst channel/fold',np.max(errs))
 print('archive wall registers',[(e['completed_campaign_index'],np.median(e['facility_telemetry_trace'],axis=0)[[3,7]].round(3).tolist()) for e in json.load(open(base+'archive_apparatus_log.json'))])
 n=len(y);idx=np.arange(n);shift=idx//len(st['sensor_ij']);rosette=idx//4
 for kind in product(models,repeat=2):
  S,U=models['P']._forward(st,op(kind));e,w=profile(S,y,idx>=0)
  es=[profile(S,y,shift!=i)[0][shift==i] for i in range(len(st['mu_shifts']))]; er=[profile(S,y,rosette%6!=i)[0][rosette%6==i] for i in range(6)];
  rms=lambda t: float(np.sqrt(np.mean(np.concatenate([np.ravel(a) for a in t])**2)))
  print(''.join(kind),'in',rms([e]),'shift cv',rms(es),[round(rms([a]),6) for a in es],'rosette cv',rms(er),max(rms([a]) for a in er),'max residual',max(abs(e)),'raw fit w',w[:S.shape[1]].round(4))
