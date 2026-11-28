import pickle, numpy as np
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['panel_c']; u=p['u']; v=p['v2']; n=len(u)
i,j=np.triu_indices(4,1)
base=np.column_stack([np.ones(n),u]); prod=u[:,i]*u[:,j]
for label, fixed in [('plain',base),('products',np.c_[base,prod]),('squares',np.c_[base,u*u]),('sin_cos',base)]:
 for family in ('cos','rational','gauss') if label!='sin_cos' else ('cos',):
  def design(h):
   q=u*h
   g={'cos':np.cos,'rational':lambda x:1/(1+x*x),'gauss':lambda x:np.exp(-x*x)}[family](q)
   return np.c_[fixed,g,np.sin(q)] if label=='sin_cos' else np.c_[fixed,g]
  def loss(h):
   a=design(h); return (a@np.linalg.lstsq(a,v,rcond=None)[0]-v).ravel()
  opts=[least_squares(loss,np.full(4,f),bounds=(.15,9),max_nfev=150) for f in (1,2,3,4,5,6)]
  o=min(opts,key=lambda s:s.cost); a=design(o.x); c=np.linalg.lstsq(a,v,rcond=None)[0]; q=np.linalg.qr(a)[0]; r=a@c-v
  print(label,family,'f',o.x.round(4),'rms',np.mean(r*r,axis=0)**.5,'press',np.mean(abs(r/(1-(q*q).sum(axis=1))[:,None]),axis=0),flush=True)
  if label in ('plain','products') and family=='cos': print(c.round(4),flush=True)
