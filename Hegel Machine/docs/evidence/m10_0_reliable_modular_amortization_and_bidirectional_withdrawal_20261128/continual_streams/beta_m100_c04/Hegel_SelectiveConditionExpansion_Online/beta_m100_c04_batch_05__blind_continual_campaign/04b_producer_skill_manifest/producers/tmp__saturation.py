import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; bank=c['single_bank_sweeps']; n=len(x)
q=np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
# Fixed redescending initialization from the gross deviation relative to the sweep cloud.
def screen(y):
 m=np.median(y,axis=0); mad=np.median(abs(y-m),axis=0); return np.max(abs(y-m)/np.maximum(mad,1e-12),axis=1)<9
print('screened rows',np.where(~screen(y))[0]); print('observation quantiles',np.quantile(s['y_obs_raw'].reshape(-1,4),[0,.5,1],axis=0));
fns={'tanh':lambda z,a:np.tanh(a*z), 'rat2':lambda z,a:z/(1+a*z*z),'sqrt':lambda z,a:z/np.sqrt(1+a*z*z),'atan':lambda z,a:np.arctan(a*z),'erf':None}
from scipy.special import erf
fns['erf']=lambda z,a:erf(a*z)
for name,fn in fns.items():
 for linear in [False,True]:
  r=np.zeros_like(y); scales=[]; coefs=[]; rb=[]
  for k in range(6):
   te=np.arange(n)%5==k if k<5 else np.ones(n,bool); tr=~te if k<5 else te.copy(); tr[tr]=screen(y[tr]); aa=[]; bb=[]
   for j in range(4):
    z=x[:,j]; A=np.c_[np.ones(n),x if linear else np.delete(x,j,axis=1),q];
    def run(a,out=False):
     D=np.c_[A,fn(z,a)]; b=np.linalg.lstsq(D[tr],y[tr,j],rcond=None)[0]; e=D[tr]@b-y[tr,j]; return (D,b) if out else e@e
    a=minimize_scalar(run,bounds=(.07,5),method='bounded').x; D,b=run(a,True); aa.append(a);bb.append(b);r[te,j]=D[te]@b-y[te,j] if k<5 else r[te,j]
    if k==5:
     mask=bank['bank_id']==j; z=bank['ref_signal'][mask]; xx=np.zeros((len(z),4));xx[:,j]=z; qq=np.column_stack([xx[:,i]*xx[:,l] for i,l in itertools.combinations_with_replacement(range(4),2)]); pred=np.c_[np.ones(len(z)),xx if linear else np.delete(xx,j,axis=1),qq,fn(z,a)]@b; e=pred-bank['raw_readout'][mask];rb.append(np.round(np.sort(abs(e))[[5,9,11]],4))
   scales.append(aa);coefs.append(bb)
  good=screen(y); z=np.linalg.norm(r,axis=1); print(name,linear,'oof p50/p95/max',np.round(np.quantile(z[good],[.5,.95,1]),5),'alpha',np.round(scales[-1],3),'bank med/2ndtail/max',np.array(rb).tolist());
  if name in ('tanh','rat2'): print('coef',np.round(coefs[-1],3))
