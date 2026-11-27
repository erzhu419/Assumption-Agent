exec(open('/tmp/chamber_reset.py').read().split('rows=[]')[0]);
print('bench/test/diag sizes',len(tr['lab_id']),len(te['lab_id']),n)
for name,panel in [('train',tr),('test',te),('base',base),('diag',g)]:
 _,bp=norm(panel); print(name,'impute column mean and cell balanced',np.nanmean(bp,axis=0),impute(bp,panel))
for name in ['diag:bayes','diag:self','base:self']:
 D=vals[name]
 for free in [False,True]:
  for link in ['fixed_hold','free_hold','separate_time','shared_gain']:
   if free and link=='separate_time':continue
   def design(q):
    tt=np.exp(q[:2]) if free else tau
    B=np.array(K(tt))[:,:2]
    target=obs-hold[:,None]*(B@th[:2]); X=D[:,None,None]*B[None,:,:]
    if link=='separate_time': X=D[:,None,None]*np.array(K(tt*np.exp(q)))[:,:2][None,:,:]
    if link=='free_hold': X=np.concatenate([X,hold[:,None,None]*B[None,:,:]],2)
    if link=='shared_gain': X=(X@(th[:2]*tt))[:,:,None]
    return X,target
   def fit(mask):
    def profile(q):
     X,y=design(q); xx=(X[mask]*w[None,:,None]).reshape(mask.sum()*len(w),X.shape[-1]); b=np.linalg.lstsq(xx,(y[mask]*w).ravel(),rcond=None)[0]; return X@b-y,b
    q=np.zeros(2)
    if free or link=='separate_time':
     q0=np.log(tau) if free else q
     q=least_squares(lambda q:(profile(q)[0][mask]*w).ravel(),q0,bounds=(q0-.4,q0+.4),max_nfev=50).x
    return *profile(q),q
   e,b,q=fit(np.ones(n,dtype=bool)); cv=np.empty_like(e)
   for l in np.unique(labs):
    mask=labs==l; cv[mask]=fit(~mask)[0][mask]
   print(name,free,link,'fit',metric(e),'cv',metric(cv),'worst-rate',max(np.sqrt(np.mean(cv[labs==l,-1]**2)) for l in np.unique(labs)),'b',np.round(b,6),'q',np.round(q,6))
