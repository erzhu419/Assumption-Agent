import itertools,pickle,numpy as np
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; x=s['ref_rosette'];y=s['raw_rosette']; q=s['single_bank_sweeps']; bank=q['bank_id'];u=q['ref_signal'];v=q['raw_readout']; xs=np.eye(4)[bank]*u[:,None]
def fit(A,Y):
 b=lstsq(A,Y)[0]
 for _ in range(22):
  r=np.abs(Y-A@b) if Y.ndim==1 else np.linalg.norm(Y-A@b,axis=1)
  w=1/np.sqrt(1+(r/max(.002,.45*np.median(r)))**2)
  b=lstsq(A*w[:,None],Y*w if Y.ndim==1 else Y*w[:,None])[0]
 return b
def powers(degree):return [i for d in range(degree+1) for i in itertools.combinations_with_replacement(range(4),d)]
def design(x,inds):return np.column_stack([x[:,i].prod(1) for i in inds])
def report(err):return np.round(np.quantile(err,[.5,.9,.95,1]),6)
for j in range(4):
 k=bank==j; a=np.polynomial.polynomial.polyvander(u[k],5)
 for d in (1,2,3,4,5):
  b=fit(a[:,:d+1],v[k]);print('single',j,d,b.round(5),report(np.sort(np.abs(v[k]-a[:,:d+1]@b))[:-1]))
keep=np.linalg.norm(y,axis=1)<8
for name,inds in [('affine',powers(1)),('cubic_diag',powers(1)+[(i,i,i) for i in range(4)]),('cubic_diag_mix',powers(2)+[(i,i,i) for i in range(4)]),('cubic',powers(3))]:
 a=design(x,inds); z=design(xs,inds); err=np.zeros_like(y); coeff=[]
 for j in range(4):
  k=bank==j; b=fit(np.r_[a[keep],z[k]],np.r_[y[keep,j],v[k]]);coeff.append(b)
  for hold in np.array_split(np.random.default_rng(1).permutation(len(x)),5):
   tr=keep.copy();tr[hold]=False; b=fit(np.r_[a[tr],z[k]],np.r_[y[tr,j],v[k]]);err[hold,j]=a[hold]@b-y[hold,j]
 r=np.linalg.norm(err[keep],axis=1); print(name,report(r),'central',report(r[np.max(abs(x[keep]),axis=1)<.65]))
 if name=='cubic':
  for i,b in zip(inds,np.array(coeff).T):print(i,b.round(5))
