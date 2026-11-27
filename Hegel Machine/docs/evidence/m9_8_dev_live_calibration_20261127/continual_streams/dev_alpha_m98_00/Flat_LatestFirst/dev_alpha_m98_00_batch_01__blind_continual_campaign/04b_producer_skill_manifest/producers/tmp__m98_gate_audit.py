exec(open('/tmp/m98_final_probe.py').read().split('A,base=data()')[0])
A,base=data()
_, _, vh = np.linalg.svd(A.reshape(-1,3),full_matrices=False)
raw=g['coupon_features_raw']; p=g['past_channel_raw']@cal[:-1]+cal[-1]
x=m._aligned_features({k:g[k][:0] for k in keys},g)[:,1:]; drift=w[2]+x@w[3:]
b0,b1=-np.expm1(-t[:,None]/ts).T
anchor=np.r_[t,1.]; b=np.c_[np.r_[b0-t/ts[0],0.],np.r_[b1-t/ts[1],0.]]
target=y-base; split=[(f'lab{v}',labs==v) for v in np.unique(labs)]+[(f'stratum{v}',strata==v) for v in np.unique(strata)]+[('missing',~complete)]
def feature(s,B): return s[:,None,None]*B[None,:,:]
F=feature(drift,b); offset=drift[:,None]*anchor
models={'unit_reset':(F,offset),'rational_time':(np.concatenate((F,feature(drift,np.r_[t/(1+(t/ts[0])**2)-t,0.][:,None])),2),offset), 'rational_drift':(np.concatenate((F,feature(drift/(1+drift**2)-drift,np.c_[anchor,b])),2),offset), 'boundary_difference':(np.concatenate((F,feature(h-p[:,-1],np.c_[np.r_[b0,1/ts[0]],np.r_[b1,1/ts[1]]])),2),offset), 'triple':(np.concatenate((F,feature(drift*h*(p[:,-1]-p[:,-2]),b)),2),offset)}
for name,(D,off) in models.items():
    scores=[]
    for label,mask in split:
        f=D[~mask].reshape(-1,D.shape[2]); c=np.linalg.lstsq(f,(target-off)[~mask].ravel(),rcond=None)[0]; scores.append(score((D@c+off-target)[mask]))
    print(name,'worst blocked',np.round(np.max(scores,0),6),'folds',np.round(scores,5).tolist())
# Use measured initial rate only as a diagnostic instrument: a large reduction points to drift inputs, not the temporal basis.
proxy=target[:,-1]; e=target-(proxy[:,None]*(anchor+b@np.array([.34880024,1.63895287]))); print('observed-rate proxy curve RMSE complete/missing',score(e[complete])[0],score(e[~complete])[0]); drerr=target[:,-1]-drift
print('rate residual means by lab (all,complete), pooled',[(int(v),float(drerr[labs==v].mean()),float(drerr[(labs==v)&complete].mean())) for v in np.unique(labs)],drerr.mean())
# An affine bridge should not leave a reproducible quadratic signature on held-out load levels.
caldata=d['calibration_load_cells']; X=caldata['bridge_raw']; Y=caldata['true_channel_loads'].sum(1)
level=np.argsort(np.argsort(Y))*5//len(Y); fields=[X,np.c_[X,*[X[:,a]*X[:,c] for a in range(4) for c in range(a,4)]]]
for f in fields:
    f=np.c_[f,np.ones(len(f))]; errs=[]
    for v in range(5):
        keep=level!=v; co=np.linalg.lstsq(f[keep],Y[keep],rcond=None)[0]; errs.append(np.sqrt(np.mean((f[~keep]@co-Y[~keep])**2)))
    print('bridge terms',f.shape[1],'worst load block RMSE',max(errs))
