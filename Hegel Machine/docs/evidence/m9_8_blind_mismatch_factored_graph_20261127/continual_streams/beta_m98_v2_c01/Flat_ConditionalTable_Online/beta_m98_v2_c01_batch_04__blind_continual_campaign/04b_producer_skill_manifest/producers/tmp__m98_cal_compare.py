import itertools,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; good=np.max(abs(y),axis=1)<9
print('excluded rows',np.flatnonzero(~good)); print(np.c_[x[~good],y[~good]])
x,y=x[good],y[good]; n=len(x)
for name in ['affine','own2','own3','own5','powers3','powers5','poly2','poly3','own3pair','powers3pair']:
    pred=np.empty_like(y); coefficients=[]; train=[]
    for j in range(4):
        cols=[np.ones(n),*x.T]
        if name.startswith('own'):
            cols.extend(x[:,j]**d for d in range(2,int(name[3])+1))
        if name.startswith('powers'):
            cols.extend(x[:,k]**d for d in range(2,int(name[6])+1) for k in range(4))
        if name.startswith('poly'):
            cols.extend(np.prod(x[:,terms],axis=1) for d in range(2,int(name[-1])+1) for terms in itertools.combinations_with_replacement(range(4),d))
        if name.endswith('pair'):
            cols.extend(x[:,k]*x[:,l] for k,l in itertools.combinations(range(4),2))
        D=np.column_stack(cols); a=np.linalg.lstsq(D,y[:,j],rcond=None)[0]; coefficients.append(a); train.append(np.std(D@a-y[:,j]));
        for i in range(n):
            tr=np.arange(n)!=i; pred[i,j]=D[i]@np.linalg.lstsq(D[tr],y[tr,j],rcond=None)[0]
    r=pred-y
    print(name,'RMSE',np.sqrt(np.mean(r*r,axis=0)).round(6),'max',np.abs(r).max().round(6),'train',np.round(train,6))
    if name in ('own3','powers3','poly3'):
        print(np.round(coefficients,5)); print('signs/large',[(mask.sum(),np.sqrt(np.mean(r[mask]**2)).round(6)) for mask in [np.max(abs(x),axis=1)<.85,np.max(abs(x),axis=1)>1.5]])
