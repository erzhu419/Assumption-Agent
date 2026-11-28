import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from transfer import *
from scipy.optimize import least_squares
m=np.isfinite(ch['coupon_features_raw']).all(1); print('incomplete',np.flatnonzero(~m));
for name,v in [('tr+ch',xc),('tr+te',aligned([tr,te],impute)),('tr',aligned([tr],impute)),('ch',aligned([ch],impute))]:
 r=theta[2]+v@theta[3:]; l=ch['lab_id']; print(name,'score complete range',r[m].min(),r[m].max());
 L=np.eye(4)[l];
 for mode,M in [('score',np.c_[np.ones(len(r)),r]),('lab offset',np.c_[L,r]),('lab gain and offset',np.c_[L,L*r[:,None]]),('lab + features',np.c_[L,v])]:
  M=M[m]; Y=y[m]; b=lstsq(M,Y)[0]; e=(Y-M@b)/(1-(np.linalg.qr(M)[0]**2).sum(1))[:,None]; print(mode,'LOO curve/rate',rms(e[:,:-1]),rms(e[:,-1]),'worst row',max(rms(e[:,:-1],1)),max(abs(e[:,-1])));
  if name=='tr+ch' and mode in ('score','lab gain and offset'):print('coeffs',lstsq(bb,b.T)[0].T.round(5))
 if name=='tr+ch':
  B=np.c_[np.ones(len(r)),r]; b=lstsq(B[m],y[m])[0]; e=y-B@b; _,s,vv=np.linalg.svd(e[m],full_matrices=False); print('singular residual',s[:6]); print('residual direction comparison',abs(vv[0]@b[1])/np.linalg.norm(b[1])); print('charge',np.c_[l,r,e[:,-1]].round(5)); print('ratios',lstsq(bb,b.T)[0].T/np.r_[theta[:2],1])
for g in [tr,te,ch]:
 print('panels per lab',[(np.median(g['ref_coupons'][g['lab_id']==j].mean(0)),np.median(g['ref_coupons'][g['lab_id']==j].std(0))) for j in range(4)])
