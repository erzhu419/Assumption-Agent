import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m98_chamber import *
from scipy.optimize import minimize_scalar
lab=diag['lab_id']; ys=y-carry@K; rs=r-carry@(1/tau)
# Fit a shared saturation kernel, profiling out individual coupon amplitudes.
def kernel(a): return np.r_[-np.expm1(-a*t)/a,1.]
def amplitudes(a, rows):
 u=kernel(a); Y=np.column_stack([ys,rs]); A=Y@u/(u@u); return A,np.mean((Y[rows]-A[rows,None]*u)**2)
fit=minimize_scalar(lambda a:amplitudes(a,np.ones(len(lab),bool))[1],bounds=(.15,3),method='bounded'); print('decay',fit.x,'rmse',np.sqrt(fit.fun)); A,_=amplitudes(fit.x,np.ones(len(lab),bool)); print('kernel 1 rmse',np.sqrt(amplitudes(1,np.ones(len(lab),bool))[1]));
keys=('lab_id','stratum','coupon_features_raw','ref_coupons'); empty={k:diag[k][:0] for k in keys}; trte={k:np.concatenate([tr[k],te[k]]) for k in keys};
for name,ref in [('diag only',empty),('train+diag',tr),('test+diag',te),('all',trte)]:
 _,x=m._features(ref,diag); d0=theta[2]+x@theta[3:]; print(name,'amp rmse',np.sqrt(np.mean((A-d0)**2)),'lab mean residual',[round(np.mean((A-d0)[lab==j]),5) for j in np.unique(lab)]); print('residual',np.round(A-d0,4))
for name,g in [('train',tr),('test',te),('diag',diag)]:
 print(name,'ref shape',g['ref_coupons'].shape)
 for j in np.unique(g['lab_id']):
  mask=g['lab_id']==j; panel=g['ref_coupons'][mask]; print(j,'raw mean',np.round(np.nanmean(g['coupon_features_raw'][mask],axis=0),3),'ref centers',np.round(np.mean(panel,axis=0),3),'ref scale',np.round(np.std(panel,axis=0),3))
