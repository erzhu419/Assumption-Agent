import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d=2):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def fit(x,y,fn,init,bounds):
 def err(p):
  a=fn(x,p);return (a@ls(a,y)-y).ravel()
 r=least_squares(err,init,bounds=bounds,max_nfev=300)
 p=r.x; a=fn(x,p);b=ls(a,y);e=a@b-y; h=np.sum(a*np.linalg.pinv(a).T,axis=1)
 return np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-h[:,None]))**2)),p,lambda z:fn(z,p)@b
funcs={
'power':(lambda x,p:aff(np.sign(x)*np.abs(x)**p[0]),[1.5],([.1],[8])),
'dead':(lambda x,p:aff(np.sign(x)*np.maximum(np.abs(x)-p[0],0)**p[1]),[.25,1.5],([0,.1],[.8,6])),
'sin':(lambda x,p:aff(np.sin(p[0]*x)),[1.5],([.05],[10])),
'sinlin':(lambda x,p:aff(np.sin(p[0]*x)+p[1]*x),[2.,.1],([.05,-5],[12,5])),
'sinsaturate':(lambda x,p:aff(np.sin(p[0]*x)/(1+p[1]*x*x)),[2.,.1],([.05,0],[12,10])),
'atan':(lambda x,p:aff(np.arctan(p[0]*x)),[2.],([.01],[30])),
'asinh':(lambda x,p:aff(np.arcsinh(p[0]*x)),[2.],([.01],[30])),
'rational':(lambda x,p:aff(x/(1+p[0]*np.abs(x)+p[1]*x*x+p[2]*abs(x)**3)+p[3]*x),[.98,.37,.05,.087],([0,0,0,-1],[20,20,20,3])),
'rational2':(lambda x,p:poly(x)/(1+p[0]*np.sum(x*x,axis=1,keepdims=True)),[.2],([0],[20])),
'rational2free':(lambda x,p:poly(x)/(1+np.sum(p*x*x,axis=1,keepdims=True)),[.2]*4,([0]*4,[20]*4)),
'ratcol':(lambda x,p:np.c_[aff(x/(1+p[0]*x*x)),x*x/(1+p[0]*x*x)],[.2],([0],[20])),
'ratlin':(lambda x,p:aff(x/(1+p[0]*x*x)+p[1]*x),[.4,.1],([0,-5],[20,5])),
'quadsep':(lambda x,p:np.c_[aff(x),x*x],[],([],[])),
}
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True,linewidth=150)
 for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  x=s[k]['u'];y=s[k][j]; print(k,j)
  q=ls(poly(x),y);print('poly coefficient rows:',q.T)
  for name,(fn,p,bounds) in funcs.items():
   if not p: continue
   rmse,loo,p,_=fit(x,y,fn,p,bounds); print(name,round(rmse,6),round(loo,6),p)
