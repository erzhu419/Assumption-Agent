exec(open('/tmp/chamber_modes.py').read().split('for yname')[0])
for mode in ('train+diag','train+test+diag','diag-only','fixed-train+test'):
 if mode=='train+diag': X=x
 elif mode=='diag-only': X=m['_aligned_features'](q,q)[n:]
 elif mode=='train+test+diag':
  joint={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; X=m['_aligned_features'](joint,q)[-n:]
 else:
  joint={k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; raw=q['coupon_features_raw'].copy()
  for l in np.unique(q['lab_id']):
   ref=joint['ref_coupons'][joint['lab_id']==l]; raw[q['lab_id']==l]=(raw[q['lab_id']==l]-np.median(ref.mean(0)))/np.median(ref.std(0))
  xj=m['_aligned_features'](tr,te)[:,1:]; labs=joint['lab_id']; st=joint['stratum']; _,ix,cnt=np.unique(np.c_[labs,st],axis=0,return_inverse=True,return_counts=True); wj=1/cnt[ix]; fill=[]
  for j in range(4):
   good=~np.isnan(joint['coupon_features_raw'][:,j]); fill.append(np.average(xj[good,j],weights=wj[good]))
  X=np.c_[np.ones(n),np.where(np.isnan(raw),fill,raw)]
 D=X@th[2:]; u=basis(tau); ones=np.ones(n); z=b[:,None]-A[:,0,:2]/u[0,:2]; Y0=Y-b[:,None]*(u[:,:2]@th[:2]); print('\nALIGN',mode)
 for name, terms in [('reset+drift-modes',[[D],[D],[D]]),('reset+affine-drift-modes',[[ones,D],[ones,D],[ones,D]]),('reset+full-X',[[*X.T],[*X.T],[*X.T]]),('reset+nonlinear-drift',[[D,D**2,D**3],[D,D**2,D**3],[D,D**2,D**3]]),('reset+residual-memory',[[D,z[:,0]],[D,z[:,1]],[D]]),('free-hold+drift',[[b,D],[b,D],[D]]),('reset+drift-modes+cross',[[D,b*D],[D,b*D],[D,b*D]])]:
  S=np.concatenate([np.array(V).T[:,None,:]*u[None,:,j,None] for j,V in enumerate(terms)],axis=-1); p=S.shape[-1]; pred=np.empty_like(Y)
  def fit(k): return np.linalg.lstsq(S[k].reshape(-1,p),Y0[k].ravel(),rcond=None)[0]
  for i in range(n): pred[i]=S[i]@fit(np.arange(n)!=i)
  e=abs(Y0-pred); print(name,'CV p90 strain/rate',np.round([np.quantile(e[:,:-1],.9),np.quantile(e[:,-1],.9)],5),'worst row strain MAE/rate',np.round([e[:,:-1].mean(1).max(),e[:,-1].max()],5),'coef',fit(slice(None)).round(4))
