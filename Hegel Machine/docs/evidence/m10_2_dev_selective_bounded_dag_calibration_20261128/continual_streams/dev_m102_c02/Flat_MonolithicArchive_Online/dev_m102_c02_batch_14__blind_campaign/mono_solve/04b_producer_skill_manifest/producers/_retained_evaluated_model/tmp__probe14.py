import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
for k in ('panel_b','panel_c'):
 u=s[k]['u']
 for l,v in s[k].items():
  if l=='u':continue
  ops=[('linear',lambda x,p:x,[],[],[]),('softplus',lambda x,p: np.logaddexp(0,p[0]*x+p[1]),[2.15,0],[.1,-5],[8,5]),('signed',lambda x,p: np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.22,1.25],[0,.15],[.7,5]),('cos',lambda x,p:np.cos(p[0]*x),[2.6],[.1],[7]),('sin',lambda x,p:np.sin(p[0]*x),[2.6],[.1],[7]),('tanh',lambda x,p:np.tanh(p[0]*x),[2.15],[.1],[8]),('rational',lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.62,1.07,1.57],[0,.1,.1],[5,5,5]),('mix',lambda x,p:np.sign(x)*abs(x)**p[0]/(1+p[1]*abs(x)**p[2]),[1.25,.22,1.25],[.1,0,.1],[5,5,5])]
  for name,op,p,lo,hi in ops:
   def resid(p):
    h=aff(op(u,p));return (h@fit(h,v)-v).ravel()
   if len(p):p=least_squares(resid,p,bounds=(lo,hi),gtol=1e-12).x
   print(k,l,name,np.round(p,6),np.sqrt(np.mean(resid(p)**2)))
u,v=[s['panel_a'][j] for j in ('u','v')];print('A ranges',u.min(),u.max(),v.min(),v.max())
for shift in (-1,1,2):
 for kind in range(3):
  w=np.roll(u,shift,axis=1);n,d=u.shape
  def block(a):return (a[:,:,None]*np.eye(d)).reshape(n*d,d)
  h=np.c_[block(np.ones_like(u)),block(u),w.ravel(),(u*w).ravel(),(u**3).ravel(),np.sin(2.4*w).ravel(),np.arctan(1.75*u).ravel()]
  if kind==1:h=h[:,:11]
  if kind==2:h=np.c_[block(np.ones_like(u)),block(u),block(np.arctan(1.75*u)),np.sin(2.4*w).ravel()]
  b=fit(h,v.ravel())
  for _ in range(35):
   e=(h@b-v.ravel()).reshape(n,d); ww=np.repeat(1/np.sqrt(.00001+np.sum(e**2,axis=1)),d); b=fit(h*ww[:,None],v.ravel()*ww)
  err=np.linalg.norm((h@b-v.ravel()).reshape(n,d),axis=1)
  print('A',shift,kind,np.round(b,5),np.quantile(err,[.25,.5,.8,.9,1]))
