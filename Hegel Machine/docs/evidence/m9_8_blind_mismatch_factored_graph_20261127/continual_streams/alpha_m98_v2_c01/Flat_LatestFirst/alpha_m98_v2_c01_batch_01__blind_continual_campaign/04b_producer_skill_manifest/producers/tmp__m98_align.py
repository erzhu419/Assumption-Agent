import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/m98_factor.py').read(),globals())
from scipy.optimize import least_squares
allg=(tr,te,dg); labs=dg['lab_id']; raw=dg['coupon_features_raw']; complete=np.all(np.isfinite(raw),axis=1)
def aligned(panel_groups, impute_groups):
    ref=np.concatenate([g['ref_coupons'] for g in panel_groups]); lab=np.concatenate([g['lab_id'] for g in panel_groups]); norms={}
    for l in np.unique(lab):
        z=ref[lab==l]; norms[l]=(np.median(z.mean(0)),np.median(z.std(0)))
    def transform(g):
        norm=np.array([norms[l] for l in g['lab_id']]); return (g['coupon_features_raw']-norm[:,0,None])/norm[:,1,None]
    xc=np.concatenate([transform(g) for g in impute_groups]); lc=np.concatenate([g['lab_id'] for g in impute_groups]); sc=np.concatenate([g['stratum'] for g in impute_groups]); _,inv,num=np.unique(np.c_[lc,sc],axis=0,return_inverse=True,return_counts=True); wt=(1/num[inv])[:,None]; mask=np.isfinite(xc); imp=(wt*np.where(mask,xc,0)).sum(0)/(wt*mask).sum(0); x=transform(dg); return np.where(np.isfinite(x),x,imp)

def kernel(mode,q):
    return np.r_[{'log':lambda:q*np.log1p(t/q),'exp':lambda:-q*np.expm1(-t/q),'rational':lambda:t/(1+t/q),'asinh':lambda:q*np.arcsinh(t/q),'power':lambda:q*((1+t/q)**.5-1)*2}[mode](),1.]
def fit(Y, a, mode, ix):
    def fn(z):return (z[1]*a[ix,None]*kernel(mode,np.exp(z[0]))-Y[ix]).ravel()
    z=least_squares(fn,[0.,1.]).x; return z, z[1]*a[:,None]*kernel(mode,np.exp(z[0]))
# First estimate the kernel without assuming any reference-panel alignment.
for mode in ('log','exp','rational','asinh','power'):
    def fn(q):
        k=kernel(mode,np.exp(q[0])); a=Y0@k/(k@k); return (a[:,None]*k-Y0).ravel()
    z=least_squares(fn,[0.]); print('free amplitude',mode,np.exp(z.x[0]),scores(z.fun.reshape(Y0.shape)).round(6))
k=kernel('log',1.); amp=Y0@k/(k@k)
for ps in ((tr,), (te,), (dg,), (tr,te),(tr,dg),(te,dg),allg):
    for ims in ((tr,te),(dg,),allg):
        x=aligned(ps,ims); a=w[2]+x@w[3:]; z,pred=fit(Y0,a,'log',np.arange(n)); cv=np.array([fit(Y0,a,'log',ix)[1][j]-Y0[j] for j,ix in enumerate(F)]); print('panels',str([len(g['lab_id']) for g in ps]),'imp',str([len(g['lab_id']) for g in ims]),'q/gain',np.r_[np.exp(z[0]),z[1]].round(5),'cv',scores(cv).round(5),'complete amp RMSE',np.sqrt(np.mean((a[complete]-amp[complete])**2)).round(5))
for l in np.unique(labs):
    ix=np.flatnonzero(labs==l); print('lab',l,'amplitude diag/required/observed:',np.c_[amp,(w[2]+aligned((tr,te),(tr,te))@w[3:]),(w[2]+aligned((dg,),(dg,))@w[3:])][ix].round(5))
