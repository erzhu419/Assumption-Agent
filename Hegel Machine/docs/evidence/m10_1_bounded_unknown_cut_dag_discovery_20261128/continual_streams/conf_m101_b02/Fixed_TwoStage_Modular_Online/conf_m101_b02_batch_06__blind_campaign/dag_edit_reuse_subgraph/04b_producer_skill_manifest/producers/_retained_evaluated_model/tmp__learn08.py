import numpy as np,pickle,sys,itertools
from scipy.optimize import least_squares,minimize_scalar
from scipy.linalg import eigh
sys.path.insert(0,'/holdout_task')
from node_00 import _inverse,_fit,_linear,_c1,_c2,_b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];z=_inverse(s['panel_a'],x);T=s['train']['traj']; y=s['train']['y']; U,S,V=np.linalg.svd(T,full_matrices=False);Y=np.c_[y,U[:,:2]*S[:2]];Y/=Y.std(0); print('target correlation',np.corrcoef(Y.T));
p=s['panel_b']; u,v=p['u'],p['v'];
for nm,fn in [('sin+tanh',lambda t,a,b:np.c_[np.sin(a*t),np.tanh(b*t)]),('sin+cubic',lambda t,a,b:np.c_[np.sin(a*t),t**3]),('sin+ratio',lambda t,a,b:np.c_[np.sin(a*t),t/(1+b*t*t)]),('sin+product',lambda t,a,b:np.c_[np.sin(a*t),t*np.roll(t,-1,axis=1)])]:
 def basis(t,w):return np.c_[_linear(t),fn(t,*w)]
 def res(w): a=basis(u,w);return (a@_fit(a,v)-v).ravel()
 q=least_squares(res,[2.8,1.2],bounds=([.2,.05],[5,5]));c=_fit(basis(u,q.x),v);print(nm,q.x,np.std(q.fun),c.round(3))
nb=len(x);folds=[np.arange(54,72),np.arange(48,72),*[np.sort(f) for f in np.array_split(np.random.default_rng(675).permutation(nb),5)]]
np.savez('/tmp/public08targets.npz',Y=Y)
# Explicit response library: assay channels and pairwise paths, with feature family scaling.
from scipy.special import erf
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import PolynomialFeatures
C=s['panel_c']; B=s['panel_b'];F=[]; labels=[];groups=[]
def add(name,a):
 a=np.asarray(a).reshape(nb,-1); F.append(a); labels.extend([f'{name}:{j}' for j in range(a.shape[1])]);groups.append((name,a))
for n,t in [('x',x),('z',z)]:
 c=_c1(t)@_fit(_c1(C['u']),C['v1']);d=_c2(t)@_fit(_c2(C['u']),C['v2']);b=_b(t)@_fit(_b(B['u']),B['v']); pn=lambda a,name:add(n+name,a)
 pn(t,'base');pn(b,'b');pn(c,'c');pn(d,'d');pn(PolynomialFeatures(4,include_bias=False).fit_transform(t),'poly')
 for j,f in [('t',t),('b',b),('c',c),('d',d)]:
  for a in [.35,.7,1.4,2.8]:
   for nm,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda w:w/(1+w*w)),('gate',lambda w:1/(1+w*w))]:pn(fun(a*f),j+nm+str(a))
  for k,g in [('t',t),('b',b),('c',c),('d',d)]:
   if j>k:continue
   fg=(f[:,:,None]*g[:,None,:]).reshape(nb,-1);pn(fg,j+k)
   for r in range(4):
    h=np.roll(g,r,axis=1)
    for a in [-1,-.5,.5,1]:
     q=f+a*h
     for nm,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda w:w/(1+w*w))]:pn(fun(q),j+k+nm+str(r)+str(a))
     pn(f/(1+(a*h)**2),j+k+'mod'+str(r)+str(a))
F=np.column_stack(F); print('features',F.shape)
def metrics(pred,te):
 pp=pred*np.c_[y,U[:,:2]*S[:2]].std(0);return [np.mean(abs(pp[:,0]-y[te])),np.mean(abs(pp[:,1:]@V[:2]-T[te]))]
for pen in [.004,.009,.018,.034,.06]:
 es=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(nb),te); m=F[tr].mean(0);sd=np.maximum(F[tr].std(0),1e-8);a=(F-m)/sd
  q=MultiTaskElasticNet(alpha=pen,l1_ratio=.82,max_iter=2200,tol=2e-4).fit(a[tr],Y[tr]); ix=np.linalg.norm(q.coef_,axis=0).argsort()[-15:][::-1];es.append(metrics(q.predict(a[te]),te));
  if len(es)==1 and pen in [.009,.034]:print('leading',pen,[(labels[i],round(np.linalg.norm(q.coef_[:,i]),3)) for i in ix])
 print('sparse',pen,np.round(es[:2],4),np.round(np.max(es,0),4),flush=True)
# Supervised nonlinear index readouts: fit the same few latent directions across the three responses.
for nm,t in [('raw',x),('inv',z)]:
 for nidx in [2,4,6]:
  es=[]
  for te in folds:
   tr=np.setdiff1d(np.arange(nb),te); rows=t[tr]; yt=Y[tr]; bas=lambda a,w:np.c_[_linear(a),np.tanh(a@w[:-1]+w[-1]),np.cos(a@w[:-1]+w[-1])];
   def res(w):
    w=w.reshape(5,nidx);f=bas(rows,w); beta=_fit(f,yt);return np.r_[(f@beta-yt).ravel(),.025*beta[5:].ravel(),.015*w.ravel()]
   qs=[least_squares(res,np.random.default_rng(k).normal(0,.55,5*nidx),max_nfev=75) for k in range(3)]; w=min(qs,key=lambda q:q.cost).x.reshape(5,nidx);es.append(metrics(bas(t[te],w)@_fit(bas(rows,w),yt),te))
  print('indices',nm,nidx,np.round(es[:2],4),np.round(np.max(es,0),4),flush=True)
