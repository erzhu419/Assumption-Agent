exec(open('/tmp/chamber_probe.py').read().split('for block,features')[0])
print('keys',list(diag),list(b)); print('labs',np.bincount(a['lab_id']),np.bincount(b['lab_id']),np.bincount(diag['lab_id']))
P,H,X=inp(diag,xd); T,Y=obs(diag); Q=np.c_[T,Y]; D=X@theta[2:]; complete=np.isfinite(diag['coupon_features_raw']).all(1)
for lab in np.unique(diag['lab_id']):
 print('panel',lab,[(round(np.median(p.mean(0)),4),round(np.median(p.std(0)),4)) for p in [z['ref_coupons'][z['lab_id']==lab] for z in (a,b,diag)]])

def ker(tau): return np.r_[-np.expm1(-t/tau),1/tau]
def history(tau):
 age=np.cumsum(dt[::-1])[::-1]-dt
 return -np.expm1(-dt/tau)*np.exp(-age/tau)

# Fit temporal and memory parameters to the measured responses jointly, avoiding ill-conditioned per-coupon amplitudes.
def run(kind,grouped=False):
 free='free' in kind; exp='exp' in kind; plus='plus' in kind; fixed='fixed' in kind; ls='ls' in kind
 def basis(p):
  tau=taus if fixed else p[:2]; m=p[2:4]; speedy=p[4]; k=ker(speedy)*speedy if exp else np.r_[t,1.]; pieces=[]
  for j in range(2):
   g=ker(tau[j]); z=P@history(m[j]);
   if free: pieces.extend(np.outer(col,g) for col in np.c_[H,P].T)
   else: pieces.extend([np.outer(H,g),np.outer(z,g)])
  pieces.append(np.outer(D,k))
  if plus: pieces.append(np.outer(D,np.r_[t,1.]))
  if ls: pieces.extend([np.outer(np.ones(len(P)),k)] + [np.outer(col,k) for col in X[:,1:].T]);
  if grouped: pieces.extend(np.outer((diag['lab_id']==l).astype(float),k) for l in np.unique(diag['lab_id'])[1:])
  return np.stack(pieces,-1)
 def optimize(ix):
  def err(p,full=False):
   B=basis(p); A=B[ix].reshape(-1,B.shape[-1]); y=Q[ix].ravel(); c=np.linalg.lstsq(A,y,rcond=1e-9)[0]; return (B@c-Q,c) if full else A@c-y
  p0=[*taus,*taus,.9]; res=least_squares(err,p0,bounds=([.2,.9,.15,.4,.2],[1.3,15,3.,15,10.]),max_nfev=150); return res.x,*err(res.x,True)
 ix=np.arange(len(Q)); p,e,c=optimize(ix); per=[]
 for train_ix,test_ix in [(ix[diag['lab_id']!=l],ix[diag['lab_id']==l]) for l in np.unique(diag['lab_id'])]:
  _,ee,_=optimize(train_ix); per.append([np.sqrt(np.mean(ee[test_ix,:-1]**2)),np.sqrt(np.mean(ee[test_ix,-1]**2))])
 print(kind,'p',p.round(4),'c',c.round(4),'fit',np.sqrt(np.mean(e[:,:-1]**2)).round(5),np.sqrt(np.mean(e[:,-1]**2)).round(5),'LOLO max',np.max(per,0).round(5),'worst rows',np.argsort(np.linalg.norm(e,axis=1))[-4:]);
 if kind in ('fixed_exp','fixed_exp_plus','fixed_exp_free','fixed_exp_ls'):
  print('rownorm complete',np.c_[np.linalg.norm(e,axis=1),complete].T.round(4)); print('memory ratios', history(p[2]).round(4),history(p[3]).round(4))
for kind in ('fixed','fixed_exp','fixed_exp_plus','fixed_exp_ls','fixed_exp_free','exp','exp_plus','exp_free'): run(kind)
