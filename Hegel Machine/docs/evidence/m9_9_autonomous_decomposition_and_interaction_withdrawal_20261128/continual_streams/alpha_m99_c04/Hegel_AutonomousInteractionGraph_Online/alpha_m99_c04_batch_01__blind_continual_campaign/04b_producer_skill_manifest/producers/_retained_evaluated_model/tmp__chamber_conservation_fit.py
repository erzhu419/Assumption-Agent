exec(open('/tmp/chamber_transfer_check.py').read().split('for physical in (False,True):')[0])
for complete in (False,True):
 use=~miss.any(1) if complete else np.ones(len(h),dtype=bool)
 f=least_squares(lambda p:((model(p,r,True)-y)[use]*w).ravel(),np.r_[np.log(q),.64,.455]); p=f.x
 print('fit',complete,'poles',np.exp(p[:2]),'load amplitude fast',p[2],'drift fast fraction',p[3],'pole scales',np.exp(p[:2])/q,'fast load fraction',(p[2]/np.exp(p[0]))/sum(b[:2]/q),'fit errors',rmse(model(p,r,True)-y))
for rule in ('complete','all'):
 pred=np.empty_like(y)
 for l in np.unique(labs):
  sel=labs==l; use=~sel & (~miss.any(1) if rule=='complete' else True)
  f=least_squares(lambda p:((model(p,r,True)-y)[use]*w).ravel(),np.r_[np.log(q),.64,.455]); pred[sel]=model(f.x,r,True)[sel]
 print(rule,'complete, missing, early, late',rmse((pred-y)[~miss.any(1)]),rmse((pred-y)[miss.any(1)]),np.sqrt(np.mean((pred[:,:4]-y[:,:4])**2)),np.sqrt(np.mean((pred[:,-5:-1]-y[:,-5:-1])**2)))
