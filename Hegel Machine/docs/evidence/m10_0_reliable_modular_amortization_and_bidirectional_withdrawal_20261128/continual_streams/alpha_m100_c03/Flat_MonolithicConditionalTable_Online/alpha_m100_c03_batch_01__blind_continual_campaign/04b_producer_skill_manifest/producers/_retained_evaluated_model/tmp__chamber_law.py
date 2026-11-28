exec(open('/tmp/chamber_probe.py').read().split('# Diagnostic reference')[0])
from scipy.optimize import least_squares
pc,hc=m._loads(c,b); y=c['strain_curves_obs']-c['strain_0'][:,None]; v=c['instant_rate_obs']; n=len(v)
Y=np.c_[y,.5*v]
xc=m._features(c,c)[:n]; ntr=len(tr['lab_id']); nt=len(te['lab_id']);
xs=m._features(dict((k,np.concatenate([tr[k],te[k]])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')),c)[ntr+nt:]

def design(xc, past=False, full=False):
    d=xc@theta[3:]; f=np.c_[np.ones(n),hc,xc if full else d]; return np.c_[f,pc] if past else f

def solve(ix,F,mode,free):
    # Both amplitude maps share the same coupon predictors. Drift can be absent,
    # constant, proportional to intrinsic drift, or an independent affine map.
    D={'none':F[:,:0], 'const':F[:,:1], 'intrinsic':np.c_[np.ones(n),xc@theta[3:]],'affine':F}[mode]
    def mat(tt):
        B=np.vstack((np.c_[-np.expm1(-t[:,None]/tt),t],.5*np.r_[1/tt,1]))
        return np.concatenate([F[:,None,:]*B[None,:,i,None] for i in range(2)]+[D[:,None,:]*B[None,:,2,None]],axis=2)
    def coeff(p):
        A=mat(np.exp(p)); a=A[ix].reshape(-1,A.shape[-1]); beta=np.linalg.lstsq(a,Y[ix].ravel(),rcond=None)[0]; return A,beta
    def fun(p):
        A,beta=coeff(p);return (A[ix]@beta-Y[ix]).ravel()
    p=np.log(tau)
    if free:p=least_squares(fun,p,bounds=(np.log([.35,1.4]),np.log([1.3,12.])),loss='soft_l1',f_scale=.003).x
    A,beta=coeff(p); a=A[ix].reshape(-1,A.shape[-1]); yy=Y[ix].ravel(); beta=least_squares(lambda q:a@q-yy,beta,jac=lambda q:a,loss='soft_l1',f_scale=.003).x
    return A@beta-Y,np.exp(p),beta

def report(e):
    e=e.copy(); e[:,-1]*=2; return np.r_[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.max(np.sqrt(np.mean(e[:,:-1]**2,axis=1))),np.max(abs(e[:,-1]))].round(5)
for label,F in [('intrinsic',design(xc)),('full',design(xc,full=True)),('pooled diag',design(xs)),('past',design(xc,past=True))]:
    for mode in ['none','const','intrinsic','affine']:
        for free in [False,True]:
            e,tt,beta=solve(np.arange(n),F,mode,free)
            cv=np.array([solve(np.delete(np.arange(n),i),F,mode,free)[0][i] for i in range(n)]);
            print(label,mode,free,'in',report(e),'loo',report(cv),'tau',tt.round(5),'coef',beta.round(5) if label=='intrinsic' else '')
