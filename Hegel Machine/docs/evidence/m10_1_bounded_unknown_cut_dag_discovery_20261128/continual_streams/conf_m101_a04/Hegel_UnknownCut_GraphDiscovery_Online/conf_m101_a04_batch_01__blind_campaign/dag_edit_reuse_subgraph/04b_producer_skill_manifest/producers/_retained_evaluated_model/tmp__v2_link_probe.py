import pickle
import numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x=s['panel_c']['u']; y=s['panel_c']['v2']; u=np.arange(len(x)); v=s['panel_c']['v1']
p=np.column_stack([np.ones(len(x))]+[x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)])
print('v1 fit noise',np.sqrt(np.mean((p@np.linalg.lstsq(p,v,rcond=None)[0]-v)**2)))
links={
 'log1p':lambda x,a:np.log1p(a*x*x)/a,
 'sqrt':lambda x,a:x*x/(1+np.sqrt(1+a*x*x)),
 'logcosh':lambda x,a:np.log(np.cosh(a*x))/a**2,
 'xtanh':lambda x,a:x*np.tanh(a*x)/a,
 'tanh2':lambda x,a:(np.tanh(a*x)/a)**2,
 'atan2':lambda x,a:(np.arctan(a*x)/a)**2,
 'asin_h2':lambda x,a:(np.arcsinh(a*x)/a)**2,
 'atan_square':lambda x,a:np.arctan(a*x*x)/a,
 'xatan':lambda x,a:x*np.arctan(a*x)/a,
 'sinc':lambda x,a:(1-np.sinc(a*x/np.pi))/a**2,
 'cos':lambda x,a:(1-np.cos(a*x))/a**2,
 'rational':lambda x,a:x*x/(1+a*x*x),
 'pseudo_huber':lambda x,a:x*x/np.sqrt(1+a*x*x),
 'sech':lambda x,a:(1-1/np.cosh(a*x))/a**2,
 'soft_exp':lambda x,a:-np.expm1(-a*x*x)/a,
}
result=[]
for name,f in links.items():
 def basis(x,a): return np.c_[np.ones(len(x)),x,f(x,a)]
 def fit(mask):
  xx=x[mask]; yy=y[mask]
  def loss(a,ret=False):
   m=basis(xx,a); w=np.linalg.lstsq(m,yy,rcond=None)[0]
   return w if ret else np.mean((m@w-yy)**2)
  opt=minimize_scalar(loss,bounds=(.01,3),method='bounded',options={'xatol':1e-10})
  return opt.x,loss(opt.x,True),np.sqrt(opt.fun)
 a,w,r=fit(u)
 e=[]; params=[]
 for i in u:
  b,ww,_=fit(u[u!=i]); e.append(basis(x[i:i+1],b)[0]@ww-y[i]); params.append(b)
 result.append((np.sqrt(np.mean(np.array(e)**2,axis=0)).max(),name,a,r,min(params),max(params)))
for z in sorted(result): print('%0.8f %-14s a=%0.5f rms=%0.8f LOO a-range=[%0.5f,%0.5f]'%z)
