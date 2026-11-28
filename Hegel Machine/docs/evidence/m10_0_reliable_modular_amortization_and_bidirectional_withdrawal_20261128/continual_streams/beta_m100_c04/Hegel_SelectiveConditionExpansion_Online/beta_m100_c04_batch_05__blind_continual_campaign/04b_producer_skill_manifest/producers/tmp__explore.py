import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; n=len(x)
print('quantiles x',np.quantile(x,[0,.1,.5,.9,1],axis=0)); print('corr',np.corrcoef(x.T)); print('raw obs',np.quantile(s['y_obs_raw'].reshape(-1,4),[0,.5,1],axis=0)); print('first ref/raw',np.c_[x,y][:8]); print('single',np.c_[*c['single_bank_sweeps'].values()])
def fit(a,b):
 w=np.ones(len(a))
 for i in range(25):
  coef=np.linalg.lstsq(a*w[:,None],b*w[:,None],rcond=None)[0]; r=np.linalg.norm(a@coef-b,axis=1); sig=max(np.median(r),1e-8); w=1/np.sqrt(1+(r/(1.8*sig))**2)
 return coef
one=np.ones((n,1)); xx=np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]); xxx=np.column_stack([x[:,i]*x[:,j]*x[:,k] for i,j,k in itertools.combinations_with_replacement(range(4),3)]); good=np.max(abs(y),axis=1)<4
for name,a in [('aff',np.c_[one,x]),('diag2',np.c_[one,x,x*x]),('poly2',np.c_[one,x,xx]),('diag3',np.c_[one,x,x*x,x**3]),('poly2+diag3',np.c_[one,x,xx,x**3]),('poly3',np.c_[one,x,xx,xxx]),('tanh',np.c_[one,x,np.tanh(x)]),('tanh+poly2',np.c_[one,x,xx,np.tanh(x)]),('rat+poly2',np.c_[one,x,xx,x/(1+x*x)]),('sin+poly2',np.c_[one,x,xx,np.sin(x)])]:
 r=np.zeros_like(y)
 for k in range(5):
  te=np.arange(n)%5==k; tr=~te; r[te]=a[te]@fit(a[tr],y[tr])-y[te]
 z=np.linalg.norm(r,axis=1); print(name,'median/p95/max',np.quantile(z[good],[.5,.95,1]),'fold worst',max(np.median(z[good&(np.arange(n)%5==k)]) for k in range(5)))
 if name in ('aff','poly2+diag3'): print('coeff',fit(a,y)); print('worst rows',sorted(zip(z,range(n)),reverse=True)[:8])
