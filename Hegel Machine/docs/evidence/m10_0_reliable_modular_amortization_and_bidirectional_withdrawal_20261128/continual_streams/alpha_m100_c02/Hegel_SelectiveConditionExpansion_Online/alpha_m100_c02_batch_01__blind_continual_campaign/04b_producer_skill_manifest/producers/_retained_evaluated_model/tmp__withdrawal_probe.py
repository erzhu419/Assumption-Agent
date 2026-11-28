exec(open('/tmp/chamber_law.py').read().split('for i,x in enumerate(xs):')[0])
print('tau theta',tau,th)
for i,x in enumerate(xs):
 drift=x@th[2:]; anchor=hold[:,None]*th[:2]; E=y-np.c_[anchor,np.zeros(len(y))]@F.T
 M=np.linalg.lstsq(F,E.T,rcond=None)[0].T
 print('pool',i,'coeff [1,r_drift] and amplitude RMS',np.linalg.lstsq(np.c_[np.ones(len(x)),drift],M,rcond=None)[0],np.sqrt(np.mean((M-np.c_[np.ones(len(x)),drift]@np.linalg.lstsq(np.c_[np.ones(len(x)),drift],M,rcond=None)[0])**2,axis=0)))
 for name,z in [('drift',drift[:,None]),('intercept+drift',np.c_[np.ones(len(x)),drift]),('feature',x),('hold+feature',np.c_[hold,x]),('history+feature',np.c_[past,hold,x])]:
  c=np.linalg.lstsq(z,M,rcond=None)[0]; res=(M-z@c)@F.T+(E-M@F.T); lev=np.sum(np.linalg.qr(z,mode='reduced')[0]**2,axis=1); loo=res/(1-lev[:,None]); print(name,'loo',np.sqrt(np.mean(loo[:,:-1]**2)),np.sqrt(np.mean(loo[:,-1]**2)), 'worst',np.max(np.sqrt(np.mean(loo[:,:-1]**2,axis=1))),np.max(abs(loo[:,-1])))
  if name=='feature':print(c)
print('feature pools lab reference centers and scales');
for q in [tr,te,g]:
 print([(round(np.median(a.mean(0)),4),round(np.median(a.std(0)),4)) for lab in np.unique(q['lab_id']) for a in [q['ref_coupons'][q['lab_id']==lab]]])
