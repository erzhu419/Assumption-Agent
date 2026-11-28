import numpy as np,pickle
from scipy.optimize import least_squares
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X,Y=map(np.asarray,(b['true_channel_loads'],b['bridge_raw'])); orig=np.arange(len(X)); clean=np.max(np.abs(Y-np.median(Y,axis=0)),axis=1)<5; X,Y,orig=X[clean],Y[clean],orig[clean]; print('excluded',np.where(~clean)[0]); o=np.array([.13,-.11,.14,-.10]); pairs=[2,3,0,1]
for j,k in enumerate(pairs):
 u,v=X[:,j],X[:,k]; y=Y[:,j];
 print('CHANNEL',j)
 def design(mode,p,u,v):
  if mode=='affine': return np.stack([np.ones_like(u),u,v],1)
  s=p[0]; t=np.tanh(s*u); base=[np.ones_like(u),t];
  if mode=='affine': return np.stack([np.ones_like(u),u,v],1)
  if mode=='premix': return np.stack([np.ones_like(u),np.tanh(s*(u+p[1]*v))],1)
  if mode=='postmix': return np.stack([np.ones_like(u),np.tanh(s*u)+p[1]*v],1)
  if mode=='rational6': base=[np.ones_like(u),u/(1+(s*u)**6)**(1/6)]
  if mode=='atan': base=[np.ones_like(u),np.arctan(s*u)]
  if mode=='erf':
   from scipy.special import erf
   base=[np.ones_like(u),erf(s*u)]
  if mode=='tanhpoly': base += [u,t*t]
  if mode in ('premix','postmix','affine'): return np.stack(base,1)
  if mode=='hinge': base += [v,abs(v),np.tanh(p[1]*v),np.abs(np.tanh(p[1]*v))]
  elif mode=='shifted': base += [v,np.tanh(p[1]*(v+p[2]))]
  elif mode=='signedpoly': base += [v,abs(v),v*v,v*abs(v),v**3]
  else: base += [v,v*v,v**3,v**4,v**5]
  return np.stack(base,1)
 for mode in ['affine','premix','postmix','tanh','tanhpoly','atan','erf','rational6','hinge','shifted','signedpoly']:
  init={'affine':[], 'premix':[1,.25], 'postmix':[1,.25],'hinge':[1,1],'shifted':[1,1,.35]}.get(mode,[1]); low={'premix':[.05,-2],'postmix':[.05,-2],'hinge':[.05,.05],'shifted':[.05,.05,-3]}.get(mode,[.05]*len(init)); high={'premix':[8,2],'postmix':[8,2],'hinge':[8,8],'shifted':[8,8,3]}.get(mode,[8]*len(init)); pred=np.zeros_like(y)
  for m in [orig%5==i for i in range(5)]+[np.zeros_like(y,dtype=bool)]:
   def ff(p,full=False):
    a=design(mode,p,u,v); q=np.linalg.lstsq(a[~m],y[~m],rcond=None)[0]; return (a@q,q) if full else (a@q-y)[~m]
   p=least_squares(ff,init,bounds=(low,high),max_nfev=250).x if init else []
   yh,q=ff(p,True); pred[m]=yh[m]
  e=abs(pred-y); full=abs(yh-y); print(mode,'OOF RMSE/max',round(np.mean(e**2)**.5,6),round(max(e),6),'full rmse',round(np.mean(full**2)**.5,6),'p,q',np.round(p,4),np.round(q,4))
 best=np.linalg.lstsq(np.stack((np.ones(len(u)),np.tanh(u)),1)[abs(v)<1e-10],y[abs(v)<1e-10],rcond=None)[0]; print('selected mixed',np.c_[u,v,y-o[j]][:22].round(4))
