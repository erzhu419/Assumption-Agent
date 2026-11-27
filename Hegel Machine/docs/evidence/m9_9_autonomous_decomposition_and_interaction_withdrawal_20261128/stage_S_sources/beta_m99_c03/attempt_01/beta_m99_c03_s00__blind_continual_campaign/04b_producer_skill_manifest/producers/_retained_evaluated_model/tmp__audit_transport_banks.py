import importlib.util,pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
S,_=m._forward(s); y=m._calibrated_observations(s); Z=s['Z_nuisance']; ij=s['sensor_ij']; ns=len(ij); groups=np.tile(ij,(len(y)//ns,1)); bank=np.asarray(s['row_bank_id']).ravel()
if len(bank)==ns: bank=np.tile(bank,len(y)//ns)
if len(bank)!=len(y): raise ValueError(f'unexpected bank mapping: {bank.shape}, {y.shape}')
labels=np.unique(bank); print('measurement bank counts:',[(int(b),int(sum(bank==b))) for b in labels])
D=(bank[:,None]==labels[None,1:]).astype(float)
for name,A in [('no nuisance',Z[:,:0]),('specified nuisance',Z),('nuisance plus bank offsets',np.c_[Z,D])]:
    e=np.zeros_like(y)
    for coord in np.unique(ij,axis=0):
        test=np.all(groups==coord,axis=1); train=~test
        a=A[train]; B=np.c_[S[train],y[train]]
        q=np.linalg.solve(a.T@a+s['nuisance_ridge']*np.eye(a.shape[1]),a.T@B)
        M=B-a@q; X,t=M[:,:-1],M[:,-1]
        w,_=m._box_simplex(X.T@X,-X.T@t,s['u_box'],s['total_budget'])
        e[test]=(S[test]-A[test]@q[:,:-1])@w-(y[test]-A[test]@q[:,-1])
    print(name,'bank RMSE:',[float(np.sqrt(np.mean(e[bank==b]**2))) for b in labels],'bank mean residual:',[float(e[bank==b].mean()) for b in labels])
c=s['sensor_calibration_sweeps']; ref=c['ref_rosette']; raw=c['raw_rosette']; B=np.linalg.lstsq(np.c_[np.ones(len(ref)),ref],raw,rcond=None)[0]; single=c['single_bank_sweeps']; ids=single['bank_id']; v=single['ref_signal']; z=single['raw_readout']; err=B[0,ids]+v*B[1+ids,ids]-z
print('affine calibration independent single-bank RMSE:',[float(np.sqrt(np.mean(err[ids==b]**2))) for b in np.unique(ids)])
