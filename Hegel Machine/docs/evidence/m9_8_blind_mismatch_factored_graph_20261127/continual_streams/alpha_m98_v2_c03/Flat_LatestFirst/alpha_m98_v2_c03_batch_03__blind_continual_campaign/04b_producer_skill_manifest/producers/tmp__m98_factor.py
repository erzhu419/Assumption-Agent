import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m98_explore import *
from scipy.optimize import least_squares
P,H,X=diag; Y=y(dg); E=F[:,:2]; r=X@th[2:]; ids=np.arange(len(Y)); labs=dg['lab_id']; W=np.r_[np.ones(len(t)),1.4]; Y0=Y-H[:,None]*(E@th[:2]);
print('diagnostic missing rows',np.argwhere(np.isnan(dg['coupon_features_raw'])).tolist());
for lab in np.unique(labs):
 re=dg['ref_coupons'][labs==lab]; rt=panels['ref_coupons'][panels['lab_id']==lab]; print('lab',lab,'count',len(re),'panel center/scale',[(np.median(rr.mean(0)),np.median(rr.std(0))) for rr in (rt,re)])
print('sweep slope/intercept per lab'); sw=cal['single_channel_sweeps']; print([np.polyfit(sw['cmd_stress'][sw['lab_id']==j],sw['true_stress'][sw['lab_id']==j],1) for j in np.unique(labs)])

def zstate(P,durs,T):
 z=np.zeros((len(P),len(T)))
 for j,dur in enumerate(durs): z+=-np.expm1(-dur/T)*(P[:,j,None]-z)
 return z

def fit(ii, drive, mem, drift='exp', exactamp=False):
 def parts(p):
  v=np.exp(p[:2]); lam=np.exp(p[2]); dr=np.r_[-np.expm1(-lam*t),lam] if drift=='exp' else np.r_[t/(1+lam*t),1.];
  Z=zstate(P,dt,taus*v)
  add=[Y0*0]
  cols=[u[:,None]*dr for u in drive.T]
  if mem=='fixed': add=[-(Z*th[:2])@E.T];
  if mem=='scaled': cols.extend([-Z[:,b,None]*E[:,b]*th[b] for b in range(2)])
  if mem=='unrestricted': cols.extend([P[:,j,None]*E[:,b] for b in range(2) for j in range(P.shape[1])])
  return add[0],np.stack(cols,-1)
 def profile(p,full=False):
  add,A=parts(p); q=(A[ii]*W[None,:,None]).reshape(-1,A.shape[-1]); yy=((Y0-add)[ii]*W).ravel(); c=np.linalg.lstsq(q,yy,rcond=None)[0]; return (add+A@c,c) if full else q@c-yy
 p=least_squares(profile,[0,0,0],bounds=(-4,4),max_nfev=100).x; yh,c=profile(p,True); return np.exp(p),c,yh-Y0
for mem in ['fixed','scaled','unrestricted']:
 for label,drive in [('intrinsic',r[:,None]),('affine',np.c_[np.ones(len(r)),r]),('separate',X)]:
  for drift in ['exp','rational']:
   p,c,e=fit(ids,drive,mem,drift); errs=[]
   for j in np.unique(labs):
    v=fit(ids[labs!=j],drive,mem,drift)[2][labs==j]; errs.append(np.sqrt(np.mean(v**2,axis=0)))
   print(mem,label,drift,'parameters',p.round(3),c.round(3),'all rmse',np.sqrt(np.mean(e**2,axis=0))[[0,7,11,12]].round(4),'LOLO max',np.max(errs,axis=0)[[0,7,11,12]].round(4));
   if mem=='fixed' and label in ['intrinsic','separate'] and drift=='exp': print('row residuals lab/r/missing/late/rate:',np.c_[labs,r,np.isnan(dg['coupon_features_raw']).any(1),e[:,11:]].round(4).tolist())
