import numpy as np,pickle
from scipy.optimize import minimize_scalar,root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def mat(*x):return np.column_stack(x)
def aff(x):return mat(np.ones(len(x)),x)
def a_design(u,j):
 x=u[:,j];w=u[:,(j+1)%4];return mat(np.ones(len(x)),x,w,x*w,x**3)
u=s['panel_a']['u'];v=s['panel_a']['v'];co=[]
for j in range(4):
 d=a_design(u,j);k=np.arange(len(u))
 for _ in range(4):
  b=ls(d[k],v[k,j]);e=abs(d@b-v[:,j]);k=np.argsort(e)[:-3]
 print('A',j,b,np.sort(e)[[0,20,38,43,44,45,46,47]],np.argsort(e)[-4:]);co.append(b)
co=np.array(co)
def zfun(points):
 def f(z):
  w=np.roll(z,-1);return np.sum(co*np.array([np.ones(4),z,w,z*w,z**3]).T,1)
 return np.array([root(lambda z:f(z)-x,x).x for x in points])
z=zfun(s['train']['x']); tr=s['train']['traj']; y=s['train']['y']; print('traj sv',np.linalg.svd(tr,compute_uv=False));print('raw z trajectory mae',np.mean(abs(aff(z)@ls(aff(z),tr)-tr)))
for panel in ['panel_b','panel_c']:
 u=s[panel]['u'];print(panel,'range',u.min(),u.max()); w=np.roll(u,-1,axis=1)
 for key,v in s[panel].items():
  if key=='u':continue
  for base in [lambda x:np.ones((len(x),1)),aff,lambda x:mat(aff(x),x*np.roll(x,-1,axis=1)),lambda x:mat(aff(x),x*x),lambda x:mat(aff(x),x**3)]:
   for name,g in [('cos',lambda x,a:np.cos(a*x)),('sin',lambda x,a:np.sin(a*x)),('rational',lambda x,a:x/(1+a*x*x)),('atan',lambda x,a:np.arctan(a*x))]:
    def d(x,a):return mat(base(x),g(x,a))
    def obj(a):
     x=d(u,a);h=np.sum(np.linalg.qr(x,mode='reduced')[0]**2,axis=1);return np.mean(abs((x@ls(x,v)-v)/(1-h[:,None])))
    aa=minimize_scalar(obj,bounds=(.15,4),method='bounded');c=d(z,aa.x)@ls(d(u,aa.x),v);print(key,base(u).shape[1],name,round(aa.x,4),round(aa.fun,6),'traj',round(np.mean(abs(aff(c)@ls(aff(c),tr)-tr)),6))
