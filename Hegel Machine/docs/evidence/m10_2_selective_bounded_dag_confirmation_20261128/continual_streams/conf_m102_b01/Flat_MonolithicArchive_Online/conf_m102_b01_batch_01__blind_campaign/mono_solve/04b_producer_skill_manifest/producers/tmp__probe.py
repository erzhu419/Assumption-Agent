import numpy as np,pickle
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def fit(u,v,trim=False):
 d=aff(u); b=ls(d,v)
 for _ in range(20 if trim else 0):
  ix=np.argsort(np.linalg.norm(v-d@b,axis=1))[:int(.85*len(u))]; b=ls(d[ix],v[ix])
 return b
for k in ['panel_a','panel_b','panel_c']:
 for name,v in s[k].items():
  if name=='u':continue
  u=s[k]['u']; b=fit(u,v,k=='panel_a');print(k,name,'affine',b,'row err quant',np.quantile(np.linalg.norm(v-aff(u)@b,axis=1),[0,.5,.8,.9,1]));print('sample',np.c_[u,v][:5]);
a=s['panel_a']; A=fit(a['u'],a['v'],True); B=fit(**s['panel_b']); c=s['panel_c']; C=[fit(c['u'],c[k]) for k in ['v1','v2']]
x=s['train']['x']; z=(x-A[0])@np.linalg.inv(A[1:]); y=s['train']['y']; t=s['train']['traj']; _,_,V=np.linalg.svd(t,full_matrices=False); Y=np.c_[y,t@V[:2].T]; print('sing uncentered',np.linalg.svd(t,compute_uv=False),'V',V[:2]); print('train vs A domains',np.quantile(z,[0,1],axis=0));
for nm,w in [('raw',x),('a',z),('braw',aff(x)@B),('b',aff(z)@B),('c1',aff(z)@C[0]),('c2',aff(z)@C[1]),('bc1',aff(aff(z)@B)@C[0]),('bc2',aff(aff(z)@B)@C[1])]:
 print('feature',nm)
 for fun in [lambda q:q,np.tanh,np.arcsinh,np.arctan,np.sin,lambda q:np.sinh(np.arcsinh(q)/3)]:
  def f(g):return aff(fun(g*w))
  for mode in [0,1]:
   def features(g):
    Z=f(g); return Z if not mode else np.c_[Z,w,*[Z[:,i]*Z[:,j] for i in range(1,5) for j in range(i,5)]]
   G=np.geomspace(.1,15,45); P=[]
   for j in range(3):
    def err(g,ret=False):
     D=features(g); pred=D@ls(D[:52],Y[:52,j]);return np.mean(abs(pred[52:]-Y[52:,j])) if ret else np.mean((pred[:52]-Y[:52,j])**2)
    g=min(G,key=err); P.append((round(g,2),round(err(g)**.5,4),round(err(g,True),4)))
   print(getattr(fun,'__name__'),mode,P)
