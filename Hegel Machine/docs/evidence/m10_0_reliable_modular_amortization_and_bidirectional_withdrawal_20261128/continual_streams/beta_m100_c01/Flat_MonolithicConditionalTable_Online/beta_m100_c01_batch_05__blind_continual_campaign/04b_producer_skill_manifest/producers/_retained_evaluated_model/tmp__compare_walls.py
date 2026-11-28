import importlib.util, pickle, numpy as np
from scipy.optimize import lsq_linear
base='/tmp/hegel_continual_archive/'
def load(i):
 p=base+'beta_m100_c01_batch_0'+str(i)+'_solver.py'; spec=importlib.util.spec_from_file_location('m'+str(i),p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
M={i:load(i) for i in [2,3,4]}
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; ref=c['ref_rosette']; raw=c['raw_rosette']; b=c['single_bank_sweeps']; x=ref; j=np.arange(4)
print('archived calibration sources:')
import inspect
print(inspect.getsource(M[4]._design),inspect.getsource(M[4]._fit_calibration))
for name,design in [('affine',lambda x,i:np.c_[np.ones(len(x)),x]),('archive',M[4]._design),('full quadratic',lambda x,i:np.column_stack([np.ones(len(x)),x]+[x[:,j]*x[:,k] for j in range(4) for k in range(j,4)]+[x[:,i]**3]))]:
 errs=[]; tests=[]
 for i in j:
  d=design(x,i); p=np.linalg.lstsq(d,raw[:,i],rcond=None)[0]; errs.append(np.sqrt(np.mean((d@p-raw[:,i])**2))); tests.append(np.sqrt(np.mean((d[::4]@np.linalg.lstsq(np.delete(d,np.arange(0,len(d),4),axis=0),np.delete(raw[:,i],np.arange(0,len(d),4)),rcond=None)[0]-raw[::4,i])**2)))
 print(name,'fit',errs,'holdout',tests)
y=M[4]._calibrate(s); z=s['Z_nuisance']; ridge=s['source_ridge']; rz=s['nuisance_ridge']; cap=s['u_box']; budget=s['total_budget']; groups=np.arange(len(y)).reshape(len(s['mu_shifts']),-1,4); allrows=np.arange(len(y))
def calc(S,train,test):
 P=np.linalg.solve(z[train].T@z[train]+rz*np.eye(z.shape[1]),z[train].T); A=S[train]-z[train]@(P@S[train]); r=y[train]-z[train]@(P@y[train]); H=A.T@A+ridge*np.eye(S.shape[1]); w,nu=M[4]._box_simplex(H,-A.T@r,cap,budget); gam=P@(y[train]-S[train]@w); e=S[test]@w+z[test]@gam-y[test]; return np.sqrt(np.mean(e**2)),w, gam
for i,m in M.items():
 S,U=m._transport(s); rms,w,g=calc(S,allrows,allrows); print('model',i,'full rms',rms,'w',w.round(5),'gamma',g.round(4));
 for name,folds in [('shift',groups),('sensor rosette',groups.transpose(1,0,2)),('edge',[np.tile((s['sensor_ij']==0).any(1),len(s['mu_shifts'])).nonzero()[0]])]:
  rr=[calc(S,np.setdiff1d(allrows,np.ravel(f)),np.ravel(f))[0] for f in folds]; print(name,'mean',np.mean(rr),'max',max(rr),'folds',np.round(rr,5))
