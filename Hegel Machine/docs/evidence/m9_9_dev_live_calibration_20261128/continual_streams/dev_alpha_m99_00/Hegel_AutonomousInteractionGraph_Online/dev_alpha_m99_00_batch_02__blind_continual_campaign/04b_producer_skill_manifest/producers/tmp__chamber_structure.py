import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
    from chamber_probe import *

def memory(p,dt,tau):
    weights=-np.expm1(-dt/tau)*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)/tau)
    return p@weights

# Apply the train/test reference alignment to the independent diagnostic coupons.
ref=np.r_[tr['ref_coupons'],te['ref_coupons']]; labs=np.r_[tr['lab_id'],te['lab_id']]
xd=g['coupon_features_raw'].copy(); xall=old._features(tr,te)
_,ii,nn=np.unique(np.column_stack((labs,np.r_[tr['stratum'],te['stratum']])),axis=0,return_inverse=True,return_counts=True)
for lab in np.unique(labs):
    r=ref[labs==lab]; mask=g['lab_id']==lab
    xd[mask]=(xd[mask]-np.median(r.mean(0)))/np.median(r.std(0))
xd=np.where(np.isfinite(xd),xd,np.average(xall,axis=0,weights=1/nn[ii]))
dr_deploy=theta[2]+xd@theta[3:]
print('drift alignment diff',np.round(dr_deploy-dr,4))

def specs(k,ts,d):
    u=np.ones(len(h)); zf=memory(p,dt,ts[0]); zs=memory(p,dt,ts[1]); zm=z[:,0]; zn=z[:,1]
    mats={
      'gap+hold,drift': ([h-zm],[h,d],[h,d]),
      'gap+hold,drift,bias': ([h-zm,u],[h,d,u],[h,d,u]),
      'retimed-gap': ([h-zf,h,d,u],[h,d,u],[h,d,u]),
      'linear-memory': ([h,zm,zn,d,u],[h,zm,zn,d,u],[h,zm,zn,d,u]),
      'retimed-memory': ([h,zf,zs,d,u],[h,zf,zs,d,u],[h,zf,zs,d,u]),
      'affine-history': ([h,*p.T,d,u],)*3,
      'nonlinear-drift': ([h,zm,zn,d,u],[h,zm,zn,d,u,d*d,h*d,h*h],[h,zm,zn,d,u,d*d,h*d,h*h]),
      'per-feature': ([h,zm,zn,*x[:,1:].T,u],)*3,
    }
    return tuple(np.column_stack(m) for m in mats[k])

def fit(k,train,d):
    weights=np.r_[np.ones(len(t)),2.]
    def design(q):
        ts=np.sort(np.exp(q)); bb=basis(ts)
        return np.concatenate([m[:,None,:]*bb[None,:,j,None] for j,m in enumerate(specs(k,ts,d))],axis=2)
    def linear(q):
        v=design(q); aa=v[train]*weights[None,:,None]; yy=y[train]*weights
        co=np.linalg.lstsq(aa.reshape(-1,v.shape[-1]),yy.ravel(),rcond=None)[0]
        return v@co,co
    res=least_squares(lambda q:((linear(q)[0][train]-y[train])*weights).ravel(),np.log(ct),bounds=(np.log([.15,.8]),np.log([.8,8.])),max_nfev=100)
    pred,co=linear(res.x)
    return pred,co,np.exp(res.x)

def rm(r):return np.sqrt(np.mean(r*r))
for label,dv in [('joint-diag',dr),('deploy-aligned',dr_deploy)]:
 for k in ['gap+hold,drift','gap+hold,drift,bias','retimed-gap','linear-memory','retimed-memory','affine-history','nonlinear-drift','per-feature']:
    pred,co,ts=fit(k,np.ones(len(h),bool),dv)
    r=pred-y; folds=[]
    for labelcol in ['lab_id','stratum']:
        for level in np.unique(g[labelcol]):
            ix=g[labelcol]==level; rr=fit(k,~ix,dv)[0][ix]-y[ix]; folds.append([rm(rr[:,:-1]),rm(rr[:,-1])])
    print(label,k,'tau',np.round(ts,5),'fit',np.round([rm(r[:,:-1]),rm(r[:,-1])],5),'worst',np.round(np.max(folds,axis=0),5),'co',np.round(co,4).tolist())
    if k=='linear-memory':
        for j in range(3):
            m=specs(k,ts,dv)[j]; aa=amps(ts)[:,j]; e=aa-m@np.linalg.lstsq(m,aa,rcond=None)[0]; print('amplitude residual',j,np.round(e,4))
print('diag-vs-test hold range',np.min(h),np.max(h),np.min(te['hold_channel_raw'].sum(axis=-1)),np.max(te['hold_channel_raw'].sum(axis=-1)))
