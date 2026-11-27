import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber06 import *
f=np.c_[F,t]@np.r_[tau*np.array([.52,.43]),.05]; observed=((y-base@F.T)@f+4*(r-base@(1/tau)))/(f@f+4)
print('lab stratum x1..4(preimputation) base drift recovered drift difference');print(np.round(np.c_[lab,g['stratum'],raw,dr,observed,observed-dr],4))
for j in np.unique(lab):
 M=labs==j; U=lab==j
 print('panel',j,[(np.median(np.nanmean(q,0)),np.median(np.nanstd(q,0))) for q in [refs[M],tr['ref_coupons'][tr['lab_id']==j],te['ref_coupons'][te['lab_id']==j],g['ref_coupons'][U]]])
 J=U&np.isfinite(raw).all(1); print('full rows gauge',np.linalg.lstsq(np.c_[np.ones(J.sum()),dr[J]],observed[J],rcond=None)[0]); print('effective raw features',np.round(g['coupon_features_raw'][U],3))
for label,xx0,lab0,st0 in [('joint',xx,labs,lst),('train',xx[:len(tr['lab_id'])],tr['lab_id'],tr['stratum']),('test',xx[len(tr['lab_id']):],te['lab_id'],te['stratum'])]:
 out=np.where(np.isnan(raw),np.array([xx0[(lab0==a)&(st0==s),1:].mean(0) for a,s in zip(lab,g['stratum'])]),raw); print(label,'cell impute residual',np.round(observed-np.c_[np.ones(len(lab)),out]@theta[2:],4));
print('lab/stratum train drift residual signed means'); C,R=m._design(tau,dt,t,ct); yt,rt=obs(tr)
for j in np.unique(tr['lab_id']):
 J=tr['lab_id']==j; print(j,np.mean((rt-R@theta)[J]),np.mean((yt-C@theta)[J],0)[-1])
