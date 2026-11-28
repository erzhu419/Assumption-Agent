import numpy as np,pickle
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; u,v,j=b['ref_signal'],b['raw_readout'],b['bank_id']; N=len(x)
def mask(y):
 m=np.median(y,axis=0); return np.max(abs(y-m)/np.maximum(np.median(abs(y-m),axis=0),1e-9),axis=1)<9
fns={'tanh':lambda z,a:[np.tanh(a*z)],'tanh+x':lambda z,a:[np.tanh(a*z),z],'tanh+x2':lambda z,a:[np.tanh(a*z),z*z],'tanh+x3':lambda z,a:[np.tanh(a*z),z**3],'tanh+x5':lambda z,a:[np.tanh(a*z),z**5],'tanh+pos3':lambda z,a:[np.tanh(a*z),np.maximum(z,0)**3],'tanh+x+x3':lambda z,a:[np.tanh(a*z),z,z**3],'rat+x':lambda z,a:[z/(1+a*z*z),z], 'atan+x':lambda z,a:[np.arctan(a*z),z], 'tanh+rat':lambda z,a:[np.tanh(a*z),z/(1+z*z)], 'tanh+hinge':lambda z,a:[np.tanh(1.1*z),np.maximum(z-a,0)**2], 'tanh+oddhinge':lambda z,a:[np.tanh(1.1*z),np.sign(z)*np.maximum(abs(z)-a,0)**2]}
for name,fn in fns.items():
 e=np.zeros_like(y); eb=np.zeros_like(v)
 for k in range(7):
  te=np.arange(N)%6==k if k<6 else np.zeros(N,bool); be=np.arange(len(u))%6==k if k<6 else np.zeros(len(u),bool); tr=~te; tr[tr]=mask(y[tr]); details=[]
  def fit(a,write=False):
   rss=0.
   for h in range(4):
    z=x[:,h]; p=x[:,(h+2)%4]; U=u[j==h]; V=v[j==h]; bt=~be[j==h]; bt[bt]=mask(V[bt,None]); A=np.column_stack([np.ones(N),p,p*p,*fn(z,a)]); B=np.column_stack([np.ones(len(U)),np.zeros(len(U)),np.zeros(len(U)),*fn(U,a)]); D=np.r_[A[tr],B[bt]]; Y=np.r_[y[tr,h],V[bt]]; C=np.linalg.lstsq(D,Y,rcond=None)[0]; d=D@C-Y; rss+=d@d
    if write:
     e[te,h]=A[te]@C-y[te,h]; eb[(j==h)&be]=B[be[j==h]]@C-V[be[j==h]]; details.append(C)
   return rss
  bounds=(.3,2.3) if 'hinge' not in name else (.25,1.4); grid=np.linspace(*bounds,15); i=np.argmin([fit(a) for a in grid]); a=minimize_scalar(fit,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded').x; fit(a,True)
 print(name,round(a,4),'rosette p50/95/max',np.round(np.quantile(np.linalg.norm(e[mask(y)],axis=1),[.5,.95,1]),5),'bank p95',round(np.quantile(abs(eb[abs(v)<4]),.95),5),'coef',np.round(np.array(details),5).tolist())
