import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; X,Y=sw['ref_rosette'],sw['raw_rosette']; bank=sw['single_bank_sweeps']
errors={k:[] for k in ('saved','shared','rational')}; terms=[]
for fold in range(8):
    test=np.arange(len(X))%8==fold
    f=m._calibration(dict(sw,ref_rosette=X[~test],raw_rosette=Y[~test])); block=[]
    for j in range(4):
        mask=bank['bank_id']==j; xb=np.zeros((mask.sum(),4)); xb[:,j]=bank['ref_signal'][mask]
        D=m._design(np.r_[X[~test],xb],j); y=np.r_[Y[~test,j],bank['raw_readout'][mask]]
        # Infer the contamination mask from training residuals, including the single-bank probes.
        e=y-D@f[j]; q=abs(e)<max(2e-4,8*1.4826*np.median(abs(e-np.median(e))))
        block.append((D[q],y[q]))
    D_test=[m._design(X[test],j) for j in range(4)]; E=np.column_stack([D@f[j] for j,D in enumerate(D_test)])-Y[test]
    valid=np.max(abs(Y[test]),axis=1)<8
    errors['saved'].append(E[valid])
    for kind in ('shared','rational'):
        init=np.r_[f[:,0],f[:,1],np.mean(f[:,2:]/f[:,1,None],axis=0)]
        if kind=='rational': init=np.r_[init,0.]
        def pred(p,D,j):
            _,a,b,c,ab,a3=D.T
            self_signal=a+p[11]*a3
            if len(p)>12: self_signal=self_signal/(1+p[12]*a*a)
            return p[j]+p[4+j]*(self_signal+p[8]*b+p[9]*c+p[10]*ab)
        def resid(p): return np.concatenate([pred(p,D,j)-y for j,(D,y) in enumerate(block)])
        bounds=(-np.inf,np.inf) if kind=='shared' else (np.r_[np.full(12,-np.inf),-.08],np.full(13,np.inf))
        p=least_squares(resid,init,bounds=bounds,gtol=1e-13).x
        errors[kind].append((np.column_stack([pred(p,D,j) for j,D in enumerate(D_test)])-Y[test])[valid]); terms.append((kind,fold,p[-1]))
for k, es in errors.items():
    print(k,'pooled RMSE',np.sqrt(np.mean(np.concatenate(es)**2)),'worst channel within fold',max(np.max(np.sqrt(np.mean(e**2,axis=0))) for e in es))
print('rational denominator coefficients',[round(t,8) for k,_,t in terms if k=='rational'])
