import importlib.util
import itertools
import pickle
import numpy as np

p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; ok=np.max(abs(y),axis=1)<8
fold=np.arange(len(x))%4
for name in ('deployed','difference','rational'):
    worst=0.
    for k in range(4):
        v=x[:,k]; prev=x[:,(k-1)%4]; nxt=x[:,(k+1)%4]; opposite=x[:,(k+2)%4]; A=a._sensor_features(x,k)
        if name=='difference': A=np.column_stack((A,opposite,(prev-nxt)**2,v*(prev-nxt),v*prev*nxt))
        if name=='rational': A=np.column_stack((A,-y[:,k]*(prev-nxt)**2,-y[:,k]*v**2))
        for f in range(4):
            train=ok&(fold!=f); test=ok&(fold==f)
            beta=np.linalg.lstsq(A[train],y[train,k],rcond=None)[0]
            if name=='rational': pred=(A[test,:-2]@beta[:-2])/(1+beta[-2]*(prev[test]-nxt[test])**2+beta[-1]*v[test]**2)
            else: pred=A[test]@beta
            worst=max(worst,np.sqrt(np.mean((pred-y[test,k])**2)))
    print('calibration',name,'worst channel-fold RMSE',worst)

# Use the existing diagnostic's independently assembled wall operators and fitting code.
ns={}; text=open('/tmp/compare_walls.py').read(); exec(text.split('ret=[]')[0],ns)
idx=ns['all_ids']; groups=ns['groups']; y=ns['y']; z=ns['z']; fit=ns['fit']; rms=ns['rms']; forward=ns['forward']
walls={bits:forward(bits) for bits in itertools.product((0,1),repeat=4)}
for adjusted in (True,False):
    records=[]
    for test in groups:
        train=np.setdiff1d(idx,test); choices=[]
        for bits,S in walls.items():
            if adjusted: w,gamma=fit(S,train); pred=S@w+z@gamma
            else:
                T=S[train]; w,_=a._box_simplex(T.T@T+1e-12*np.eye(T.shape[1]),-T.T@y[train],s['u_box'],s['total_budget']); pred=S@w
            choices.append((rms((pred-y)[train]),bits,rms((pred-y)[test])))
        records.append(min(choices))
    print('outer-fold wall selection; nuisance adjusted',adjusted,'states',sorted({r[1] for r in records}),'worst RMSE',max(r[2] for r in records))
