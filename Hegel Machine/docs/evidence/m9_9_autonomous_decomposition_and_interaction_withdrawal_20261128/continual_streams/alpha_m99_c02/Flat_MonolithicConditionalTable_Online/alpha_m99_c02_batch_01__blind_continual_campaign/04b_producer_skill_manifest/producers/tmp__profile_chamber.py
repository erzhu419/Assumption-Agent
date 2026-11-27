import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
    from probe_m99 import *
from scipy.optimize import least_squares
p,h=m._loads(g,b); labs=g['lab_id']; n=len(h)
_, gx0=m._features(tr,g)
print('features missing counts',np.isnan(g['coupon_features_raw']).sum(0),'max feature alignment change',np.max(abs(gx-gx0),axis=0))
Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]
def time(ta):
    return np.vstack((np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1.]))
def mem(widths):
    z=np.zeros((n,len(widths)))
    for delta, stress in zip(dt,p.T):
        z+=(-np.expm1(-delta/widths))*(stress[:,None]-z)
    return z
D=th[2]+gx@th[3:]; M=mem(ta); F0=np.c_[np.ones(n),h,M,D]; F1=np.c_[np.ones(n),h,p,gx]
def err(e): return np.r_[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))]
for name,F in [('physical',F0),('independent main effects',F1)]:
 for lab in [None,*np.unique(labs)]:
    fit=np.ones(n,bool) if lab is None else labs!=lab; out=fit if lab is None else ~fit
    def pred(tau):
        B=time(tau); coeff=np.linalg.lstsq(F[fit],Y[fit],rcond=None)[0]@np.linalg.pinv(B.T)
        return F@coeff@B.T,coeff
    opt=least_squares(lambda a:(pred(np.exp(a))[0][fit]-Y[fit]).ravel(),np.log(ta),bounds=(np.log([.08,.5]),np.log([3,25])))
    tuned=np.exp(opt.x)
    for tau in (ta,tuned):
        yp,C=pred(tau); print(name,lab,'tau',tau,'rmse',err((yp-Y)[out]));
        if lab is None: print(C)
# The baseline-subtracted response is structurally sparse even when its coefficient estimates are noisy.
B=time(ta); c,r=m._basis(ta,p,h,gx,dt,t); E=Y-np.c_[c@th,r@th]; Z=E@np.linalg.pinv(B.T)
for F in (F0,F1):
 C=np.linalg.lstsq(F,Z,rcond=None)[0]; print('correction affine coefficients\n',C); print('correction component unexplained std',np.std(Z-F@C,axis=0))
