import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d=2):return np.c_[np.ones(len(x)),*[np.prod(x[:,ii],1) for j in range(1,d+1) for ii in itertools.combinations_with_replacement(range(x.shape[1]),j)]]
def prof(x,y,make,p,b=(-np.inf,np.inf)):
 def fun(p):
  f=make(x,p);return (f@ls(f,y)-y).ravel()
 pp=least_squares(fun,p,bounds=b,max_nfev=120).x;w=ls(make(x,pp),y);return pp,w,np.linalg.norm(fun(pp))/np.sqrt(y.size)
a=s['panel_a'];keep=np.linalg.norm(a['v']-a['u'],axis=1)<4; aw=ls(aff(a['u'][keep]),a['v'][keep]); cal=lambda x:(x-aw[0])/aw[1:].diagonal()
c=s['panel_c']; c1=lambda x,p:aff(np.arctan(p[0]*x+p[1:])); pc,wc,e=prof(c['u'],c['v1'],c1,[2.15,0,0,0,0]);print('c1',pc,e,wc)
for panel,key in [('b','v'),('c','v2')]:
 x,y=s['panel_'+panel]['u'],s['panel_'+panel][key]; print(panel,key,'poly2 coeff',ls(poly(x),y));
 tests={'poly2':(lambda x,p:poly(x),[1]),'poly2_atan':(lambda x,p:poly(np.arctan(p[0]*x)),[2.15]),'poly2_sin':(lambda x,p:poly(np.sin(p[0]*x)),[1.5]),'poly2_tanh':(lambda x,p:poly(np.tanh(p[0]*x)),[1.]),'poly2_ratsquash':(lambda x,p:poly(x/np.sqrt(1+p[0]**2*x*x)),[.6]),'poly2_via_c1':(lambda x,p:poly(c1(x,pc)@wc),[1]),'poly2_via_c1input':(lambda x,p:poly(np.arctan(pc[0]*x+pc[1:])),[1]),'radial':(lambda x,p:poly(x)/(1+p[0]**2*np.sum(x*x,1,keepdims=True)),[.4])}
 for k,(f,p) in tests.items():
  pp,w,e=prof(x,y,f,p); X=f(x,pp);lev=np.sum(np.linalg.svd(X,full_matrices=False)[0]**2,axis=1);print(k,pp,e,np.sqrt(np.mean(((X@w-y)/(1-lev[:,None]))**2)))
t=s['eval_times']; tr=s['train'];z=tr['traj']; y=tr['y']; x=tr['x']; q=cal(x)
for label,fn in [('exp+t',lambda p:np.c_[1-np.exp(-p[0]*t),t]),('exp+exp',lambda p:np.c_[1-np.exp(-p[0]*t),1-np.exp(-p[1]*t)]),('exp+log',lambda p:np.c_[1-np.exp(-p[0]*t),np.log1p(p[1]*t)]),('exp+sin',lambda p:np.c_[1-np.exp(-p[0]*t),np.sin(p[1]*t)]),('exp+rat',lambda p:np.c_[1-np.exp(-p[0]*t),t/(1+p[1]*t)])]:
 p,w,e=prof(t,z.T,lambda _,p:fn(p),[1.4,.4]);print('time',label,p,e)
u,d,v=np.linalg.svd(z,full_matrices=False); out=np.c_[y,z@v[:2].T];n=52
for label,h in [('x',x),('q',q),('atanq',np.arctan(pc[0]*q+pc[1:])),('c1q',c1(q,pc)@wc),('c1raw',c1(x,pc)@wc),('c2q',poly(q)@ls(poly(c['u']),c['v2'])),('c2raw',poly(x)@ls(poly(c['u']),c['v2']))]:
 for d in (1,2,3):
  F=poly(h,d);w=ls(F[:n],out[:n]);print(label,d,np.mean(abs(F[n:]@w-out[n:]),0));
