exec(open('/tmp/chamber_probe.py').read().split('# Regress each component')[0])
from scipy.optimize import minimize
D=v@w[2:]; P=np.c_[h,p]; idx=np.arange(len(h)); L=q['lab_id']; S=q['stratum']; ONE=np.ones_like(h); BL=np.eye(4)[L][:,1:]; BS=np.eye(3)[S][:,1:]
# Fit all times and the initial derivative jointly, rather than regressing poorly conditioned projected amplitudes.
def fit_linear(z, blocks, use, baseline):
 B=basis(z); F=np.concatenate([J[:,None,:]*B[None,:,j,None] for j,J in enumerate(blocks)],2)
 a=F[use].reshape(-1,F.shape[-1]); b=(y-baseline)[use].ravel()
 beta=np.linalg.lstsq(a,b,rcond=None)[0]; return baseline+F@beta,beta
C=np.c_[A@w,R@w]; Z=np.empty((len(h),0)); V=np.c_[ONE,D]; G=R[:,:2]*tau; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
def state(z):return p@(-np.expm1(-dt[:,None]/z)*np.exp(-tail[:,None]/z))
def model(name,z):
 if name=='affine':return [np.c_[ONE,P,x],np.c_[ONE,P,x],np.c_[ONE,P,x]],np.zeros_like(y)
 if name=='affine_lab':return [np.c_[ONE,P,x,BL]]*3,np.zeros_like(y)
 if name=='pulses_drift':return [np.c_[ONE,p,D],np.c_[ONE,p,D],V],(h[:,None]*w[:2])@basis(z)[:,:2].T
 if name=='pulses_features':return [np.c_[p,v],np.c_[p,v],v],(h[:,None]*w[:2])@basis(z)[:,:2].T
 if name=='pulses_drift_lab':return [np.c_[p,V,BL],np.c_[p,V,BL],np.c_[V,BL]],(h[:,None]*w[:2])@basis(z)[:,:2].T
 if name=='pulses_drift_stratum':return [np.c_[p,V,BS],np.c_[p,V,BS],np.c_[V,BS]],(h[:,None]*w[:2])@basis(z)[:,:2].T
 if name=='common_state':return [np.c_[state(z),V]]*2+[V],(h[:,None]*w[:2])@basis(z)[:,:2].T
 if name=='gap_drift':return [np.c_[h,state(z),V]]*2+[V],np.zeros_like(y)
for name in ['affine','affine_lab','pulses_drift','pulses_features','pulses_drift_lab','pulses_drift_stratum','common_state','gap_drift']:
 def pred(z,ii):
  blocks,base=model(name,z); return fit_linear(z,blocks,ii,base)
 def opt(ii):
  return least_squares(lambda z:(pred(z,ii)[0][ii]-y[ii]).ravel(),tau,bounds=([.3,1.5],[1.4,16]),max_nfev=60).x
 for free in (False,True):
  z=opt(idx) if free else tau; f,b=pred(z,idx)
  errs=[]
  for groups in (idx,L,S):
   e=np.zeros_like(y)
   for j in np.unique(groups):
    m=groups==j; ii=idx[~m]; e[m]=(pred(opt(ii) if free else tau,ii)[0]-y)[m]
   errs.append(np.r_[np.sqrt(np.mean(e[:,:-1]**2)),np.max(np.sqrt(np.mean(e[:,:-1]**2,axis=0))),np.sqrt(np.mean(e[:,-1]**2)),np.max(abs(e[:,-1]))])
  print(name,'free',free,'tau',z.round(4),'train',np.sqrt(np.mean((f-y)**2)).round(5),'LOO/LOlab/LOstratum',np.array(errs).round(4).tolist());
  if free and name in ('affine','pulses_drift','pulses_features','common_state'): print('  coef',b.round(5))
print('train/test keys',list(tr), list(te)); print('diagnostic panel stats vs pooled',[(l,[(np.median(a.mean(0)),np.median(a.std(0))) for a in (ref[labs==l],q['ref_coupons'][L==l])]) for l in np.unique(L)])
