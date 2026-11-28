import pickle, sys, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m100_01_batch_04_solver as prev
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; X,Y=sw['ref_rosette'],sw['raw_rosette']; bank=sw['single_bank_sweeps']
np.set_printoptions(precision=5,suppress=True,linewidth=160)
print('sweep schema',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in sw.items()]); print('multibank first 12',np.c_[X[:12],Y[:12]]); print('single bank 0',np.c_[bank['ref_signal'][:12],bank['raw_readout'][:12]])
curves={'tanh':lambda a,k: np.tanh(k*a)/k,'atan':lambda a,k: np.arctan(k*a)/k,'rational':lambda a,k: a/(1+(k*a)**2),'sqrt':lambda a,k: a/np.sqrt(1+(k*a)**2),'erf':lambda a,k: __import__('scipy.special',fromlist=['erf']).erf(k*a)*np.sqrt(np.pi)/(2*k),'exp':lambda a,k: np.sign(a)*(-np.expm1(-k*np.abs(a)))/k}
for j in range(4):
 use=bank['bank_id']==j; u=bank['ref_signal'][use]; v=bank['raw_readout'][use]; a=X[:,j]; others=X[:,np.arange(4)!=j]; old=prev._calibration(sw)[j]; results=[]
 for name,f in curves.items():
  for linear in (False,True):
   def eval(x,p): return p[0]+p[1]*f(x,np.exp(p[2]))+(p[3]*x if linear else 0)
   def fit(mask):
    q=[old[0],old[1],-.15]+([0.] if linear else []); return least_squares(lambda p: eval(u[mask],p)-v[mask],q,loss='soft_l1',f_scale=.003,max_nfev=220,bounds=([-1,-20,-3]+([-20] if linear else []),[1,20,2]+([20] if linear else []))).x
   e=np.array([eval(u[i],fit(np.arange(len(u))!=i))-v[i] for i in range(len(u))]); q=fit(np.ones(len(u),bool)); results.append((np.quantile(abs(e),.8),name,linear,q,e))
 results.sort(key=lambda r:r[0]); print('bank',j,'isolated LOOCV ranked',[(name,lin,round(err,6)) for err,name,lin,_,_ in results]); best=results[0]; _,name,lin,q,e=best; f=curves[name]; d=Y[:,j]-(q[0]+q[1]*f(a,np.exp(q[2]))+(q[3]*a if lin else 0)); print('best',q,'isolated LOO residuals',e,'full residual quantiles',np.quantile(d,[0,.1,.25,.5,.75,.9,1])); print('full residual first 12',d[:12]); print('full residual corr (a, others, a*others)',np.corrcoef(np.c_[np.clip(d,*np.quantile(d,[.1,.9])),a,others,a[:,None]*others].T)[0,1:]);
 for label,D in [('additive',others),('pairwise',np.c_[others,a[:,None]*others,others[:,0]*others[:,1],others[:,0]*others[:,2],others[:,1]*others[:,2]]),('mixed saturation',np.c_[others,others*f(a,np.exp(q[2]))[:,None],others*(a*a)[:,None],others*others*a[:,None]])]:
  D=np.c_[np.ones(len(a)),D]; errs=[]
  for k in range(5):
   train=np.arange(len(a))%5!=k; p=least_squares(lambda p:D[train]@p-d[train],np.zeros(D.shape[1]),loss='soft_l1',f_scale=.01).x; errs.extend((D[~train]@p-d[~train]).tolist())
  print(label,'held-out abs p50/p90',np.quantile(np.abs(errs),[.5,.9]))
