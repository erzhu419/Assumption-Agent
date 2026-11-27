import pickle,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; np.set_printoptions(precision=5,suppress=True)
print('active coordinates:',np.count_nonzero(x,axis=1).tolist()); print('large raw cells:',np.argwhere(abs(y)>4).tolist())
terms=[t for d in range(4) for t in itertools.combinations_with_replacement(range(4),d)]
D=np.column_stack([np.prod(x[:,t],axis=1) for t in terms]); fold=np.arange(len(x))%5
for label,idx in [('own-cubic+mixed-affine', [i for i,t in enumerate(terms) if len(set(t))<=1 and len(t)!=2]), ('quadratic+own-cubic', [i for i,t in enumerate(terms) if len(t)<=2 or len(set(t))==1]), ('quadratic+own-cubic+triples',[i for i,t in enumerate(terms) if len(t)<=2 or len(set(t)) in (1,3)]), ('full-cubic',list(range(len(terms))))]:
 A=D[:,idx]; O=np.empty_like(y); fits=[]
 for k in range(4):
  for f in range(6):
   m=fold!=f; a=A[m]; b=y[m,k]; fit=least_squares(lambda q:a@q-b,np.zeros(a.shape[1]),jac=lambda q:a,loss='soft_l1',f_scale=.015,max_nfev=160).x
   for j in range(3):
    good=np.abs(a@fit-b)<.12
    if good.sum()>=a.shape[1]: fit=np.linalg.lstsq(a[good],b[good],rcond=None)[0]
   if f<5: O[~m,k]=y[~m,k]-A[~m]@fit
  fits.append(fit)
 print(label,'OOF median',np.median(abs(O),axis=0),'90th',np.quantile(abs(O),.9,axis=0),'fraction > .12',(abs(O)>.12).mean(0))
 if label=='full-cubic':
  for k,q in enumerate(fits):
   r=A@q-y[:,k]; print('channel',k,'coef',[(terms[i],round(float(v),5)) for i,v in zip(idx,q) if abs(v)>.003],'outliers',[(int(j),round(float(r[j]),3)) for j in np.flatnonzero(abs(r)>.12)])
