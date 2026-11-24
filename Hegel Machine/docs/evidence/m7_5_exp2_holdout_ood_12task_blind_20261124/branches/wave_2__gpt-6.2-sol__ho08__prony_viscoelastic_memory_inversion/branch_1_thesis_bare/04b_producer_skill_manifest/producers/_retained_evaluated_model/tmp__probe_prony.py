import numpy as np,pickle
from scipy.optimize import nnls
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']
def design(t,x,tau):
 z=np.diff(t)[:,None]/tau; a=np.exp(-z); f=-np.expm1(-z)/z; h=np.zeros((len(t),len(tau)))
 for i,d in enumerate(np.diff(x)): h[i+1]=a[i]*h[i]+f[i]*d
 return np.column_stack((x,h))
def fit(A,D,y,w):
 w=np.sqrt(w); Aw=A*w[:,None]; Dw=D*w[:,None]; yw=y*w
 Q,_=np.linalg.qr(Dw); b,_=nnls(Aw-Q@(Q.T@Aw),yw-Q@(Q.T@yw),maxiter=4000); d=np.linalg.lstsq(Dw,yw-Aw@b,rcond=None)[0]
 return b,d,y-A@b-D@d

def run(e,tau,verbose=False):
 t=e['train_t']; y=e['train_stress_raw']; D=np.column_stack((np.ones(len(t)),(t-t.mean())/np.ptp(t))); A=design(t,e['train_strain'],tau)
 w=np.ones(len(t)); results={}
 for j in range(24):
  b,d,r=fit(A,D,y,w); sc=1.4826*np.median(np.abs(r-np.median(r))); sc=max(sc,1e-12); w=np.minimum(1,1.345*sc/np.maximum(np.abs(r),1e-15))
 results['hub']=b; results['sc']=sc
 if verbose: print('drift',d,'resid quantiles',np.quantile(np.abs(r)/sc,[.5,.9,.95,.97,.975,.98,.985,.99,1]))
 for clip in [3,3.5,4,4.5,5,6]:
  wc=(np.abs(r)<=clip*sc).astype(float); bc,dc,rc=fit(A,D,y,wc); results[clip]=bc
 return results
if __name__=='__main__':
 for n in [41,81,121]:
  tau=np.geomspace(.02,20,n); out=run(e,tau,True); C=design(e['test_t'],e['test_strain'],tau); truth=p['gt_test_stress']
  for k,b in out.items():
   if k=='sc': print('noise',b); continue
   print(n,k,'G',b[0], 'err',np.linalg.norm(C@b-truth)/np.linalg.norm(truth),'modes',list(zip(tau[b[1:]>0].round(3),b[1:][b[1:]>0].round(2))))
