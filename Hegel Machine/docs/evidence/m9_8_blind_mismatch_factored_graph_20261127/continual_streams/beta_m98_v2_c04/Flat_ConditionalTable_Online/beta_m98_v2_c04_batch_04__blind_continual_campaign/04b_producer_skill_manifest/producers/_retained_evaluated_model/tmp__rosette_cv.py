import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; x,y=a['ref_rosette'],a['raw_rosette']; np.set_printoptions(precision=6,suppress=True,linewidth=160)
print('single-axis count',np.unique(np.count_nonzero(x,axis=1),return_counts=True))
def features(x,j,k):
 cols=[np.ones(len(x)),*x.T]
 if k=='diag3': cols.append(x[:,j]**3)
 if k=='all3': cols.extend((x**3).T)
 if k=='full3': cols.extend(np.prod(x[:,ids],axis=1) for d in (2,3) for ids in itertools.combinations_with_replacement(range(4),d))
 return np.column_stack(cols)
def fit(d,y,k,n):
 mask=np.ones(len(y),bool)
 for _ in range(9):
  c=np.linalg.lstsq(d[mask],y[mask],rcond=None)[0]; r=abs(y-d@c); sel=np.argsort(r)[:max(d.shape[1]+2,len(y)-n)]; m=np.zeros(len(y),bool); m[sel]=True
  if np.array_equal(m,mask): break
  mask=m
 return c
for k in ('affine','diag3','all3','full3'):
 print('model',k)
 for j in range(4):
  d=features(x,j,k); errors=np.zeros(len(y));
  for test in np.array_split(np.random.default_rng(42).permutation(len(y)),5):
   train=np.setdiff1d(np.arange(len(y)),test); c=fit(d[train],y[train,j],k,7); errors[test]=y[test,j]-d[test]@c
  c=fit(d,y[:,j],k,8); r=y[:,j]-d@c; out=np.argsort(abs(r))[-8:]; print(j,'OOF abs quantiles',np.quantile(abs(errors),[.5,.8,.9,.95,1]),'coef',c if k!='full3' else '', 'largest residuals',list(zip(out,r[out].round(5))))
b=a['single_bank_sweeps']; print('bank-only robust cubic fits');
for j in range(4):
 v=b['bank_id']==j; u=b['ref_signal'][v]; q=b['raw_readout'][v]; d=np.c_[np.ones(len(u)),u,u**3]; c=fit(d,q,'',1); print(j,c,q-d@c)
