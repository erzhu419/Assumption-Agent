exec(open('/tmp/rheo_diag.py').read().split("for k,z,xx")[0])
from scipy.optimize import least_squares
p,h=m._loads(q,b); y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; lab=q['lab_id']; labs=np.unique(lab)
B=np.r_[-np.expm1(-t[:,None]/tau),(1/tau)[None,:]].T
T=np.r_[t,1.]; hold=(h[:,None]*th[:2])@B
mem=h[:,None]-m._design(xq,p,h,dt,t,tau)[1][:,:2]*tau
M=mem*th[:2]; dr=lambda xx:th[2]+xx@th[3:]
variants={'joint3':xq,'diag_only':m._features(q,q)[:len(h)],'train_diag':m._features(tr,q)[n:],'test_diag':m._features(te,q)[len(te['lab_id']):]}
# Apply the public train/test reference and imputation map without refitting it on diagnostics.
laball=np.r_[tr['lab_id'],te['lab_id']]; rs=np.vstack((tr['ref_coupons'],te['ref_coupons'])); xx=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw'])).copy(); qq=q['coupon_features_raw'].copy()
for g in labs:
 panel=rs[laball==g]; mid=np.nanmedian(np.nanmean(panel,0)); sd=np.nanmedian(np.nanstd(panel,0)); xx[laball==g]=(xx[laball==g]-mid)/sd; qq[lab==g]=(qq[lab==g]-mid)/sd
_,cells,counts=np.unique(np.c_[laball,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); w=1/counts[cells,None]; fill=np.nansum(w*xx,axis=0)/np.sum(w*~np.isnan(xx),axis=0); variants['joint2_map']=np.where(np.isnan(qq),fill,qq)
clean=~np.isnan(q['coupon_features_raw']).any(1)
print('nominal drift from different reference sets:'); print(np.c_[*[dr(v) for v in variants.values()],clean].round(4));
for name,xx in variants.items():
 D=np.c_[np.ones(len(h)),dr(xx)]; coef=np.linalg.lstsq(D,y-hold,rcond=None)[0]; print(name,'fit slope by time then rate:',coef[1].round(4),'intercept',coef[0].round(4),'RMSE',np.sqrt(np.mean((y-hold-D@coef)**2)))

def ex(k): return np.r_[-np.expm1(-t/k),1/k]
def expint(k): return T-k*ex(k)
def rational(k): return np.r_[t/(1+t/k),1.]
def tanh(k): return np.r_[k*np.tanh(t/k),1.]
def gauss(k):
 from scipy.special import erf
 return np.r_[k*np.sqrt(np.pi)/2*erf(t/k),1.]
def log(k): return np.r_[k*np.log1p(t/k),1.]

def run(features, form, f, addmem=False, intercept=False):
 R=dr(features); shapepar=form not in ('branch','linear');
 def design(k):
  b=np.atleast_2d(f(k)); F=[R[:,None]*v for v in b];
  if intercept: F += [np.broadcast_to(v,y.shape) for v in b]
  if addmem: F += [M[:,j,None]*B[j] for j in range(2)]
  return np.stack(F,-1)
 def fit(mask):
  def inner(k):
   a=design(k); mat=a[mask].reshape(-1,a.shape[-1]); c=np.linalg.lstsq(mat,(y-hold)[mask].ravel(),rcond=None)[0]; return (hold+a@c-y),c
  k=1.
  if shapepar: k=np.exp(least_squares(lambda z:inner(np.exp(z[0]))[0][mask].ravel(),[0.],bounds=([-3],[4])).x[0]);
  return *inner(k),k
 E,c,k=fit(np.ones(len(h),bool)); errors=np.zeros_like(E)
 for g in labs:
  s=lab==g; errors[s]=fit(~s)[0][s]
 rms=lambda z: np.sqrt(np.mean(z**2));
 return np.round([rms(E),max(rms(errors[lab==g,:-1]) for g in labs),max(rms(errors[lab==g,-1]) for g in labs),max(rms(errors[:,j]) for j in range(len(t))),rms(E[clean]),k],5),c.round(5)
forms=[('linear',lambda k:T),('branch',lambda k:np.r_[B,T[None,:]]),('ex',ex),('ex+T',lambda k:[ex(k),T]),('exint',expint),('rational',rational),('tanh',tanh),('gauss',gauss),('log',log)]
for name,xx in variants.items():
 print(name,'metrics: fit overall, worst lab curve, worst lab rate, worst horizon, fit complete rows, shape parameter');
 for form,f in forms:
  print(form,run(xx,form,f))
print('joint2 refinements with fitted residual memory and drift offset')
for form,f in forms[1:]:
 for addmem,offset in [(True,False),(False,True),(True,True)]:print(form,addmem,offset,run(variants['joint2_map'],form,f,addmem,offset))
