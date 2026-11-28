import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X,Y=map(np.asarray,(c['true_channel_loads'],c['bridge_raw'])); idx=np.arange(len(X)); keep=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<5; X,Y,idx=X[keep],Y[keep],idx[keep]; pairs=[2,3,0,1]
for j,k in enumerate(pairs):
 u,v,y=X[:,j],X[:,k],Y[:,j]; print('channel',j)
 def basis(mode,p):
  s,t=p[:2]; ones=np.ones_like(u); main=np.tanh(s*u); cross=np.tanh(t*v); base=[ones,v,cross,v*v/(1+(t*v)**2)]
  if mode=='add': return np.array(base+[main,u,u**3]).T
  if mode=='rational': return np.array(base+[u/(1+(s*u)**2),u,u**3]).T
  if mode=='sqrt': return np.array(base+[u/np.sqrt(1+(s*u)**2),u,u**3]).T
  if mode=='atan': return np.array(base+[np.arctan(s*u),u,u**3]).T
  if mode=='erf': return np.array(base+[erf(s*u),u,u**3]).T
  if mode=='sin': return np.array(base+[np.sin(s*u),u,u**3]).T
  if mode=='tanh2': return np.array([ones,v,cross,cross**2,main,u,u**3]).T
  if mode=='mix': return np.array(base+[main,u,u**3,np.tanh(s*(u+p[2]*v))]).T
  if mode=='mix2': return np.array(base+[main,u,u**3,np.tanh(s*(u+p[2]*v)),np.tanh(s*(u-p[2]*v))]).T
  if mode=='interaction': return np.array(base+[main,u,u**3,main*v,main*v*v,main**2*v,main**2*v*v,u*v]).T
  if mode=='poly': return np.array([u**a*v**b for a in range(8) for b in range(7) if a+b<=7]).T
 for mode in ['add','rational','sqrt','atan','erf','sin','tanh2','mix','mix2','interaction','poly']:
  err=np.empty(len(u));
  for m in [idx%5==i for i in range(5)]+[idx<0]:
   def evaluate(p):
    A=basis(mode,p); q=np.linalg.lstsq(A[~m],y[~m],rcond=None)[0]; return A@q-y,q
   if mode in ('mix','mix2'): seeds=[[1,1,h] for h in [-1,-.3,.3,1]]; lo=[.1,.1,-3];hi=[5,5,3]
   else: seeds=[[.8,.8],[1.5,1.5],[2.5,2.5]];lo=[.1,.1];hi=[5,5]
   fit=min((least_squares(lambda p:evaluate(p)[0][~m],s,bounds=(lo,hi),max_nfev=160) for s in seeds),key=lambda r:r.cost); e,q=evaluate(fit.x); err[m]=e[m]
  print(mode,'oof/max/full',*[f'{f:.6g}' for f in (np.sqrt(np.mean(err**2)),max(abs(err)),np.sqrt(np.mean(e**2)))], 'p',fit.x.round(3),'coef',q.round(4) if len(q)<15 else '', 'corr',np.round(np.corrcoef([e,u*v,u*v*v,u*u*v,u*u*v*v])[0,1:],2))
