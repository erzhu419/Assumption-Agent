import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe07 import *
from scipy.optimize import least_squares
from numpy.linalg import lstsq
print('panel conventions: lab, train/test (center, scale), diag (center, scale)')
for lab in np.unique(labs):
 def ms(a): return np.array([np.median(a.mean(0)), np.median(a.std(0))])
 print(lab,ms(refs[labs==lab]),ms(g['ref_coupons'][g['lab_id']==lab]))
base=np.c_[(h[:,None]-z)*th[:2], np.zeros(len(h))]@basis.T
Xs={'train/test':x,'diag':m._features(g,g)[:len(h),1:],'diag with train':m._features(tr,g)[len(H):,1:]}
for align,xx in Xs.items():
 dd=np.c_[np.ones(len(h)),xx]@th[2:]; one=np.ones(len(h))
 for name,W in [('drift',dd[:,None]),('offset',np.c_[dd,one]),('quadratic',np.c_[dd,dd**2,one]),('hold',np.c_[dd,h,one]),('memory',np.c_[dd,h,z,one]),('feature',np.c_[dd,xx,one])]:
  full=(W[:,:,None,None]*basis.T[None,None,:,:]).transpose(0,3,1,2).reshape(len(h),len(t)+1,-1)
  def fit(f):
   a=full[f].reshape(-1,full.shape[-1]); b=(v-base)[f].ravel(); start=lstsq(a,b,rcond=None)[0]; return least_squares(lambda c:a@c-b,start,jac=lambda c:a,loss='soft_l1',f_scale=.009).x
  err=np.zeros_like(v)
  for lab in np.unique(g['lab_id']):
   f=mask&(g['lab_id']!=lab); e=g['lab_id']==lab; err[e]=(base+full@fit(f)-v)[e];
  print(align,name,'coef',fit(mask).round(4),'LOLO',np.round([max(np.sqrt(np.mean(err[mask&(g['lab_id']==lab),a]**2)) for lab in np.unique(g['lab_id'])) for a in [slice(None,-1),-1]],4),'p50,90,100',np.quantile(np.sqrt(np.mean(err[mask]**2,axis=1)),[.5,.9,1]).round(4))
  if name=='drift': print('coupon err',np.c_[dd, err[:,-2:]].round(4).tolist())
