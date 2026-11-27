import pickle,itertools,numpy as np
from scipy.optimize import least_squares
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=st['sensor_calibration_sweeps']; x,y=[np.asarray(sw[k]) for k in ('ref_rosette','raw_rosette')]; b=sw['single_bank_sweeps']; eye=np.eye(4); pairs=list(itertools.combinations_with_replacement(range(4),2)); triples=list(itertools.combinations_with_replacement(range(4),3)); A=np.c_[np.ones(len(x)),x]; Q=np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); C=np.column_stack([x[:,i]*x[:,j]*x[:,k] for i,j,k in triples]); z0=np.linalg.lstsq(A,y,rcond=None)[0]; w=np.ones(len(x))
for step in range(15):
 z0=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]; r=np.linalg.norm(y-A@z0,axis=1); w=1/np.sqrt(1+(r/max(.25,np.median(r)))**2)
print('affine coeffs',np.round(z0,4)); print('largest residual rows',[(int(i),round(r[i],3)) for i in np.argsort(r)[-15:]]); clean=r<max(2.5,5*np.median(r)); print('clean rows',sum(clean)); print('bank rows',np.round(np.c_[b['bank_id'],b['ref_signal'],b['raw_readout']],4)); print('axis raw',np.round(y[:16],4))
starts=[np.zeros(4)]+list(eye*.7)+[.5*(eye[i]+s*eye[j]) for i,j in itertools.combinations(range(4),2) for s in [-1,1]]
for typ in ['affine','quad','cubic','q_over','lq_over','qdiag_over','full_over']:
 rr=np.zeros_like(y);
 for j in range(4):
  if typ in ['affine','quad','cubic']:
   D={'affine':A,'quad':np.c_[A,Q],'cubic':np.c_[A,Q,C]}[typ];
   for k in range(4):
    va=np.arange(len(x))%4==k; tr=clean&~va; cf=np.linalg.lstsq(D[tr],y[tr,j],rcond=None)[0]; rr[va,j]=D[va]@cf-y[va,j]
   continue
  def design(q):
   d=(1+(x@q)**2)[:,None];
   if typ=='q_over': return np.c_[A,Q/d]
   if typ=='qdiag_over': return np.c_[A,Q/(1+(x[:,j]*q[0])**2)[:,None]]
   if typ=='lq_over': return np.c_[A[:,0],x/d,Q/d]
   return np.c_[A,(np.c_[x,Q])/d]
  for k in range(5):
   va=np.arange(len(x))%4==k; tr=clean&~va; target=y[tr,j];
   def fit(q,coef=False):
    D=design(q)[tr]; c=np.linalg.lstsq(D,target,rcond=1e-10)[0]; return c if coef else D@c-target
   pre=sorted(starts[1:] if typ=='full_over' else starts,key=lambda q:np.linalg.norm(fit(q)))[:5]; op=min((least_squares(fit,s,bounds=(-4,4),max_nfev=100) for s in pre),key=lambda op:op.cost)
   if k==4:
    print(typ,j,'q',np.round(op.x,3),'fit',round(np.sqrt(2*op.cost/tr.sum()),6),'coeff',np.round(fit(op.x,True),3)); continue
   rr[va,j]=design(op.x)[va]@fit(op.x,True)-y[va,j]
 print('OOF',typ,'by channel',np.round(np.sqrt(np.mean(rr[clean]**2,axis=0)),5),'axis',np.sqrt(np.mean(rr[:16][clean[:16]]**2)),'max clean',np.max(np.abs(rr[clean])))
