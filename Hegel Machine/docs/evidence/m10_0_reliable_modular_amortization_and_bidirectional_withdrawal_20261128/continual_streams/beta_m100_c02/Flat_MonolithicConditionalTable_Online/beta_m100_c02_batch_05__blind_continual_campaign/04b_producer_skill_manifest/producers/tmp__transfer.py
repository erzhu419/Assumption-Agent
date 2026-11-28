import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; X,Y=a['ref_rosette'],a['raw_rosette']; b=a['single_bank_sweeps']; np.set_printoptions(precision=7,suppress=True,linewidth=140)
funcs={'tanh':np.tanh,'atan':np.arctan,'erf':erf,'sqrt':lambda z:z/np.sqrt(1+z*z),'rational':lambda z:z/(1+z*z)}
for i in range(4):
 j=(b['bank_id']==i)&(abs(b['raw_readout'])<5); t=b['ref_signal'][j]; y=b['raw_readout'][j]; sel=abs(Y[:,i])<5; x=X[sel]; v=Y[sel,i]; results=[]
 for name,f in funcs.items():
  for lin in (False,True):
   def predict(p,z): return p[0]+p[1]*f(p[2]*z)+(p[3]*z if lin else 0.)
   p=least_squares(lambda p:predict(p,t)-y,[0,1,1]+([.1] if lin else []),max_nfev=1000).x; e=predict(p,t)-y
   results.append((np.max(abs(e)),name,lin,p))
 print('bank',i,sorted(results,key=lambda a:a[0])[:6]); _,name,lin,p=min(results,key=lambda a:a[0]); f=funcs[name]
 r=v-predict(p,x[:,i]); print('residual quantiles',np.quantile(r,[0,.1,.5,.9,1]));
 terms=[z for k in range(1,4) for z in combinations_with_replacement(range(4),k) if any(q!=i for q in z)]; A=np.stack([np.prod(x[:,z],axis=1) for z in terms],1); c=np.linalg.lstsq(A,r,rcond=None)[0]; print('residual poly',np.quantile(abs(r-A@c),[.5,.95,1]),sorted(zip(terms,c),key=lambda u:-abs(u[1]))[:12]);
 for inds in ((i+2,), tuple(j for j in range(4) if j!=i)):
  B=x[:,inds];
  for mode in ('pre','post','both'):
   def mod(q):
    if mode=='pre': return predict(p,x[:,i]+B@q)
    if mode=='post': return predict(p,x[:,i])+B@q
    return predict(p,x[:,i]+B@q[:len(inds)])+B@q[len(inds):]
   q=least_squares(lambda q: mod(q)-v,np.zeros(len(inds)*(2 if mode=='both' else 1))).x; print(inds,mode,q,np.quantile(abs(mod(q)-v),[.5,.95,1]))
