from rheo_cv import *
from scipy.linalg import null_space

def profile(q):
    y,r=target(q); n=len(r); tt=np.r_[t,.7]; Y=np.c_[y,.7*r]; P=null_space(tt[None,:]); yp=Y@P
    def mat(z):
        A,B=designs(q,np.exp(z),'z'); return np.einsum('ntj,tk->nkj',np.concatenate([A[:,:,:2],.7*B[:,None,:2]],axis=1),P)
    o=least_squares(lambda par:(mat(par[:2])@par[2:]-yp).ravel(),np.r_[np.log([.68,4]),[.65,.5]],bounds=([np.log(.0501)]*2+[0]*2,[np.log(24.99)]*2+[np.inf]*2),loss='cauchy',f_scale=.009,max_nfev=180)
    tau=np.exp(o.x[:2]); A,B=designs(q,tau,'z'); response=Y-o.x[2:][None,None,:].__mul__(np.concatenate([A[:,:,:2],.7*B[:,None,:2]],axis=1)).sum(2)
    dd=np.median(np.c_[response[:,:len(t)]/t,response[:,-1]/.7],axis=1); x,past,hold=prep(q); lab=q['lab_id']; modes=np.c_[np.ones(n),x]; known=np.isfinite(q['coupon_features_raw']).all(1); f=fit2(q,q,.004)
    print('projected taus & weights',tau,o.x[2:],'p90 orthogonal',np.quantile(abs(o.fun),.90))
    drift_error=dd-modes@f[1][2:]
    for l in np.unique(lab):
        sel=(lab==l)&known; e=drift_error[sel]; X=modes[sel]; print('lab',l,'n',len(e),'drift error quantiles',np.round(np.quantile(e,[.1,.5,.9]),5),'drift coeff',np.round(np.linalg.lstsq(X,dd[sel],rcond=None)[0],5))
    dsg=np.c_[modes,hold,past, q.get('aux_chamber_telemetry',np.zeros((n,0)))]; print('drift correlations',np.round(np.corrcoef(np.c_[drift_error[known],dsg[known,1:]],rowvar=False)[0,1:],3))

for name,q in [('bench',tr),('chamber',ch)]:
    print(name); profile(q)
    n=len(q['lab_id']); x,past,hold=prep(q)
    seen=[]; E=[]; R=[]; F=[]
    for j in range(n):
        mask=np.arange(n)!=j; qs=cut(q,mask); qt=cut(q,~mask)
        # Reference panels and the imputation mean use covariates only; compute once from the full cohort.
        ref=q; y,r=target(qs)
        def matrices(z):
            tau=np.exp(z); A,B=designs(q,tau,'z',ref); return A,B
        f=fit2(q,q,.004)
        def res(z):
            A,B=matrices(z[:2]); return np.r_[((A[mask]@z[2:])-y).ravel(),.7*(B[mask]@z[2:]-r)]
        o=least_squares(res,np.r_[np.log(f[0]),f[1]],bounds=([np.log(.0501)]*2+[-np.inf]*7,[np.log(24.99)]*2+[np.inf]*7),loss='soft_l1',f_scale=.004,max_nfev=100)
        A,B=matrices(o.x[:2]); yt,rt=target(qt); E.append((A[j]@o.x[2:])-yt[0]); R.append((B[j]@o.x[2:])-rt[0]); F.append(np.exp(o.x[:2]));
    E,R,F=np.array(E),np.array(R),np.array(F); print('label-LOO pooled',summ(E,R),'tau range',np.round(np.quantile(F,[0,.5,1],axis=0),4));
    for lbl,s in [(str(l),q['lab_id']==l) for l in np.unique(q['lab_id'])]+[('complete',np.isfinite(q['coupon_features_raw']).all(1)),('missing',~np.isfinite(q['coupon_features_raw']).all(1))]: print('slice',lbl,summ(E[s],R[s]))
