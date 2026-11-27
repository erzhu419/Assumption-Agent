import itertools,pickle,numpy as np
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']
print('calibration nonzero counts',np.unique(np.count_nonzero(X,axis=1),return_counts=True)); print('zero rows',Y[np.all(X==0,axis=1)])
def design(x,name):
 cols=[np.ones(len(x)),*x.T]
 for degree in range(2,5):
  if str(degree) in name: cols.extend(x[:,j]**degree for j in range(4))
  if f'c{degree}' in name: cols.extend(np.prod(x[:,js],axis=1) for js in itertools.combinations_with_replacement(range(4),degree) if len(set(js))>1)
 return np.stack(cols,axis=1)
def fit(a,y):
 b=np.linalg.lstsq(a,y,rcond=None)[0]
 for k in range(32):
  r=np.linalg.norm(y-a@b,axis=1); sw=1/np.hypot(1,r/max(.0001,2*np.median(r)))
  b=np.linalg.lstsq(a*sw[:,None],y*sw[:,None],rcond=None)[0]
 return b
names=['linear','2','3','23','2c2','2c23','2c23c3','2c234']
perm=np.random.default_rng(21).permutation(len(X)); folds=np.array_split(perm,8)
base=np.zeros_like(Y); a=design(X,'2c23');
for te in folds:
 tr=np.setdiff1d(perm,te); base[te]=Y[te]-a[te]@fit(a[tr],Y[tr])
spike=np.linalg.norm(base,axis=1)>1
print('spike rows',np.flatnonzero(spike).tolist(), 'remaining', (~spike).sum())
for name in names:
 a=design(X,name); oof=np.zeros_like(Y)
 for te in folds:
  tr=np.setdiff1d(perm,te); oof[te]=Y[te]-a[te]@fit(a[tr],Y[tr])
 print(name,a.shape[1], 'clean rmse',np.round(np.sqrt(np.mean(oof[~spike]**2,axis=0)),6),'norm p50 p90 max',np.round(np.quantile(np.linalg.norm(oof[~spike],axis=1),[.5,.9,1]),6), 'axis/mixed rmse',[round(float(np.sqrt(np.mean(oof[m&~spike]**2))),6) for m in (np.count_nonzero(X,axis=1)<2,np.count_nonzero(X,axis=1)>1)])
 if name in ('2c23','2c23c3'): print('coefficients\n',np.round(fit(a,Y),5))
