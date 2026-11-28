exec(open('/tmp/chamber_probe.py').read().split('for drift in')[0])
labs=d['lab_id']; E=np.r_[t,1.]; xx=m._features(d,d)[:nd]; r=th[2]+xx@th[3:]; F=np.c_[h,r,np.ones(nd)]; Z=h[:,None]-c[:,0,:2]/(-np.expm1(-t[0]/tau));
def fit(tr, kind, qfree):
 f=F[:,:2] if kind!='offset' else F
 if kind=='mem': f=np.c_[f,Z]
 def prof(q):
  V=np.vstack([et(a) for a in q]+[E]); B=np.zeros((f.shape[1],3)); B[0,:2]=th[:2];
  if kind=='locked':
   B[1]=np.linalg.lstsq(V.T, (r[tr] @ (y[tr]-(f@B@V)[tr]))/(r[tr]@r[tr]), rcond=None)[0]
  else:
   B=np.linalg.lstsq(V.T,np.linalg.lstsq(f[tr],y[tr],rcond=None)[0].T,rcond=None)[0].T
   if kind in ('plain','mem'):
    A=np.einsum('ni,jk->nkij',f,V).reshape(len(y),len(E),-1); ids=np.arange(B.size)!=2; B.ravel()[2]=0; B.ravel()[ids]=np.linalg.lstsq(A[tr][:,:,ids].reshape(-1,sum(ids)),y[tr].ravel(),rcond=None)[0]
  return f@B@V,B
 q=tau if not qfree else least_squares(lambda q:(prof(q)[0][tr]-y[tr]).ravel(),tau,bounds=([.5,2.5],[.9,5.5])).x
 Y,B=prof(q);return Y,B,q
for k in ('locked','plain','offset','mem'):
 for fr in (False,True):
  out=np.empty_like(y)
  for lab in np.unique(labs):
   tr=labs!=lab; out[~tr]=fit(tr,k,fr)[0][~tr]
  est,B,q=fit(np.ones(nd,dtype=bool),k,fr); ee=out-y; score=lambda e: [np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))]; print(k,fr,q,np.round(B,6), 'full',score(est-y),'OOF',score(ee),'worst',np.max([score(ee[labs==l]) for l in np.unique(labs)],axis=0))
  if k=='plain' and not fr: np.save('/tmp/chamber_amplitude.npy',B)
