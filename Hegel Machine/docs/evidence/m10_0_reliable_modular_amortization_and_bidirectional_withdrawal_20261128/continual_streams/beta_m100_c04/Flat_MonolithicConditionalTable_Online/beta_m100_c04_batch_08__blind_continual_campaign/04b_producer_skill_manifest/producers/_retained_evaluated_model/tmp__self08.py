import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; b=s['single_bank_sweeps']; i,t,z=[b[k] for k in ('bank_id','ref_signal','raw_readout')]; ok=np.abs(z)<8;i,t,z=i[ok],t[ok],z[ok];
forms={'tanh':np.tanh,'atan':np.arctan,'erf':erf,'sin':np.sin,'sqrt':lambda x:x/np.sqrt(1+x*x),'ratio':lambda x:x/(1+x*x)}
for name,f in forms.items():
 for kind in ['linear','cubic','shift','linear_shift','cubic_linear']:
  def design(t,i,q):
   a,p=q; v=f(a*t+(p if 'shift' in kind else 0)); v=v+(p*t**3 if kind=='cubic' else 0); return np.column_stack((np.eye(4)[i],np.eye(4)[i]*v[:,None],*[np.eye(4)[i]*(t**j)[:,None] for j in ([1,3] if kind=='cubic_linear' else [1] if 'linear' in kind else [])]))
  def res(q,tr,va):
   A=design(t[tr],i[tr],q); c=np.linalg.lstsq(A,z[tr],rcond=None)[0];return design(t[va],i[va],q)@c-z[va]
  ix=np.arange(len(z)); fold=np.tile(np.arange(12)%3,4); scores=[]
  for k in range(3):
   tr=ix[fold!=k];va=ix[fold==k]; candidates=[least_squares(lambda q:res(q,tr,tr),[a,0.],bounds=([.15,-3],[4,3]),gtol=1e-13) for a in [.5,1.,2.]]; r=min(candidates,key=lambda r:r.cost); scores.append(np.sqrt(np.mean(res(r.x,tr,va)**2)))
  print(f'{name:5s} {kind:13s} {max(scores):.8g} {r.x}')
