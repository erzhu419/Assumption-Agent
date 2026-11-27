import importlib.util, pickle
from itertools import combinations, combinations_with_replacement
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(p+'public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; clean=np.max(abs(y),axis=1)<7
fold=np.arange(len(x))%5
terms=[j for d in range(4) for j in combinations_with_replacement(range(4),d)]
full=np.stack([np.prod(x[:,j],axis=1) for j in terms],axis=1)
print('clean rows',clean.sum(),'faulted rows',np.flatnonzero(~clean).tolist(), 'largest clean / smallest fault amplitude',np.max(abs(y[clean])),np.min(np.max(abs(y[~clean]),axis=1)))
for family in ['sparse cubic','full cubic','pairwise difference cubes','rational']:
    errors=[]
    for f in range(5):
        train=clean&(fold!=f); test=clean&(fold==f); pred=[]
        for i in range(4):
            X=m._features(x,i); candidates=[(X,np.ones(len(x)))]
            if family=='full cubic': candidates=[(full,np.ones(len(x)))]
            if family=='pairwise difference cubes':
                diffs=np.column_stack([(x[:,j]+sign*x[:,k])**3 for j,k in combinations(range(4),2) for sign in (-1,1)])
                candidates=[(np.c_[X,diffs],np.ones(len(x)))]
            if family=='rational':
                Q=[x[:,j] for j in range(4)]+[x[:,j]+sign*x[:,k] for j,k in combinations(range(4),2) for sign in (-1,1)]
                candidates += [(X,1+a*q*q) for q in Q for a in (1e-4,1e-3,1e-2,0.1,1.)]
            if len(candidates)>1:
                def loss(cand):
                    A,den=cand; A=A/den[:,None]; vals=[]
                    for g in range(5):
                        if g==f: continue
                        a=train&(fold!=g); b=train&(fold==g); c=np.linalg.lstsq(A[a],y[a,i],rcond=None)[0]; vals.extend((A[b]@c-y[b,i])**2)
                    return np.mean(vals)
                chosen=min(candidates,key=loss)
            else: chosen=candidates[0]
            A,den=chosen; A=A/den[:,None]; c=np.linalg.lstsq(A[train],y[train,i],rcond=None)[0]; pred.append(A[test]@c)
        errors.append(np.sqrt(np.mean((np.column_stack(pred)-y[test])**2,axis=0)))
    print(family,'worst fold-channel RMSE',np.max(errors))
b=sw['single_bank_sweeps']; idx=b['bank_id'].astype(int); bx=np.eye(4)[idx]*b['ref_signal'][:,None]; c=m._rosette_coefficients(x,y); by=np.array([m._features(bx[j],i)@c[i] for j,i in enumerate(idx)]); ok=abs(b['raw_readout'])<7
print('independent single-bank RMSE by channel', [float(np.sqrt(np.mean((by[ok&(idx==i)]-b['raw_readout'][ok&(idx==i)])**2))) for i in range(4)])
S,_=m._forward(s); ycal=m._calibrate(s); Z=s['Z_nuisance']; shifts=np.repeat(np.arange(len(s['mu_shifts'])),len(s['sensor_ij'])); I=np.eye(Z.shape[1]); R=s['source_ridge']*np.eye(S.shape[1]); cap=s['u_box']; budget=s['total_budget']
for use_z in (False,True):
    rms=[]
    for k in np.unique(shifts):
        tr=shifts!=k; z=Z[tr]; P=np.linalg.solve(z.T@z+s['nuisance_ridge']*I,z.T) if use_z else np.zeros((I.shape[0],len(z))); D=S[tr]-z@(P@S[tr]); t=ycal[tr]-z@(P@ycal[tr]); w,_=m._box_budget(D.T@D+R,-D.T@t,cap,budget); e=S[~tr]@w+Z[~tr]@(P@(ycal[tr]-S[tr]@w))-ycal[~tr]; rms.append(np.sqrt(np.mean(e**2)))
    print('held shift RMSE by shift, nuisance profiling',use_z,rms)
