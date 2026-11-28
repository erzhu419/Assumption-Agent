exec(open('/tmp/chamber_probe.py').read().split('# Regress each component')[0])
from scipy.optimize import least_squares
L=q['lab_id']; S=q['stratum']; ids=np.arange(len(h)); rawq=q['coupon_features_raw']; X={}
for name,parts in [('deploy',(tr,te)),('train',(tr,)),('diagnostic',(q,)),('train_diag',(tr,q)),('all',(tr,te,q))]:
    rr=np.vstack([a['ref_coupons'] for a in parts]); ll=np.concatenate([a['lab_id'] for a in parts]); qx=rawq.copy()
    for l in np.unique(L):
        r=rr[ll==l]; qx[L==l]=(qx[L==l]-np.median(r.mean(0)))/np.median(r.std(0))
    llq,iv,ct=np.unique(np.c_[L,S],axis=0,return_inverse=True,return_counts=True)
    ww=(1/ct[iv])[:,None]; mq=np.nansum(ww*qx,axis=0)/np.sum(ww*np.isfinite(qx),axis=0)
    X[name]=(np.where(np.isfinite(qx),qx,mean),np.where(np.isfinite(qx),qx,mq))
print('missing diag',np.argwhere(~np.isfinite(rawq)).tolist()); print('r ranges',[(k,np.round((a@w[3:]+w[2]).min(),3),np.round((a@w[3:]+w[2]).max(),3)) for k,(a,b) in X.items()])

def eval_model(qx, mode):
    D=qx@w[3:]+w[2]; one=np.ones(len(h)); base=np.c_[one,h,D]; poly=np.c_[base,D**2,D**3,D*h,h**2]; labs_=np.eye(4)[L][:,1:]
    J={'drift':base,'curved':poly,'raw':np.c_[one,h,qx], 'lab':np.c_[base,labs_], 'lab_gain':np.c_[base,labs_,labs_*D[:,None]], 'stress':np.c_[base,p], 'smooth':np.c_[base,np.tanh(2*D),np.maximum(D,0)],'amplitudes':base}[mode]
    K=np.c_[one,D] if mode=='amplitudes' else J
    def fit(z, ii):
        B=basis(z); F=np.concatenate((B[None,:,0,None]*J[:,None,:],B[None,:,1,None]*J[:,None,:],B[None,:,2,None]*K[:,None,:]),2)
        b=np.linalg.lstsq(F[ii].reshape(-1,F.shape[-1]),y[ii].ravel(),rcond=None)[0]
        return F@b, b
    def opt(ii):
        z=least_squares(lambda z:(fit(z,ii)[0][ii]-y[ii]).ravel(),[.712,4.8],bounds=([.5,2.5],[1.,10]),max_nfev=35).x
        return z
    z=opt(ids); f,b=fit(z,ids); cv=[]
    for groups in (ids,L,S):
        e=np.empty_like(y)
        for j in np.unique(groups):
            m=groups==j; ii=ids[~m]; e[m]=(fit(opt(ii),ii)[0]-y)[m]
        cv.extend([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))])
    return np.sqrt(np.mean((f-y)**2)),z,cv,b
results=[]
for panel,(a,b) in X.items():
    for imp,qx in enumerate((a,b)):
        if imp and np.array_equal(a,b):continue
        for mode in ('drift','amplitudes','curved','raw','lab','lab_gain','stress','smooth'):
            rms,z,cv,beta=eval_model(qx,mode); results.append((max(cv),panel,imp,mode,rms,z,cv,beta))
            if mode in ('drift','amplitudes'): print(panel,imp,mode,round(rms,5),z.round(4),np.round(cv,5),beta.round(4))
for _,panel,imp,mode,rms,z,cv,beta in sorted(results,key=lambda t:t[0])[:12]:print('top',panel,imp,mode,round(rms,5),z.round(4),np.round(cv,5),beta.round(4))
