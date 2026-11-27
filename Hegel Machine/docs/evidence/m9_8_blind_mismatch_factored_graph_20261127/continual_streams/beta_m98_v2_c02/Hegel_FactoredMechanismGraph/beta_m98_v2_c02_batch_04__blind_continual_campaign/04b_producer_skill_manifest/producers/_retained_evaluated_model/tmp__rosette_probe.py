import pickle, numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; bank=c['single_bank_sweeps']
for j in range(4):
 m=bank['bank_id']==j; print('bank',j,np.round(np.c_[bank['ref_signal'][m],bank['raw_readout'][m]],5).tolist())
def design(x,mode):
 a=[np.ones(len(x)),*x.T]; d=int(mode) if mode.isdigit() else 0
 if d:
  for k in range(2,d+1): a.extend(x[:,ii].prod(1) for ii in itertools.combinations_with_replacement(range(4),k))
 if mode=='diag': a.extend(x.T**2); a.extend(x.T**3); a.extend(x.T**5)
 if mode=='pair': a.extend(x.T**2); a.extend(x.T**3); a.extend(x[:,ii].prod(1) for ii in itertools.combinations(range(4),2))
 return np.array(a).T
def fit(A,Y):
 w=np.ones(len(A))
 for i in range(35):
  sw=np.sqrt(w)[:,None]; co=np.linalg.lstsq(A*sw,Y*sw,rcond=None)[0]; r=np.linalg.norm(Y-A@co,axis=1); w=1/(1+(r/max(.002,1.8*np.median(r)))**2)
 return co
if __name__=='__main__':
 r0=np.linalg.norm(y-design(x,'3')@fit(design(x,'3'),y),axis=1); clean=r0<.20; print('excluded rows',np.where(~clean)[0],np.round(r0[~clean],2)); print('clean cubic residual quantiles',np.quantile(r0[clean],[0,.5,.9,1])); radius=np.linalg.norm(x,axis=1); center=np.max(np.abs(x),axis=1)<.85; edge=radius>=np.quantile(radius,.75); fold=np.random.default_rng(62).permutation(len(x))%8
 for name in ('1','2','3','4','diag','pair'):
  A=design(x,name); yp=np.empty_like(y)
  for k in range(8):
   t=fold!=k; yp[~t]=A[~t]@fit(A[t],y[t]);
  r=np.abs(yp-y); g=lambda m: np.round(np.sqrt(np.mean(r[m&clean]**2,axis=0)),5); print(name, g(np.ones(len(x),bool)),g(center),g(edge))
 print('cubic coefficients rows', np.round(fit(design(x,'3'),y),4))
