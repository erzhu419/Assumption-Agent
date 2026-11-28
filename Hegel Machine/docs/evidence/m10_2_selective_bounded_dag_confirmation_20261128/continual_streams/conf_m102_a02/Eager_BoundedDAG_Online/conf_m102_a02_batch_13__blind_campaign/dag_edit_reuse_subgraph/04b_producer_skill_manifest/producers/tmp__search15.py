import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.column_stack([s['train']['y'],s['train']['traj']]);
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
def cal(p,x):
 u,v=p['u'],p['v']; w=np.roll(u,-1,axis=1); C=[]
 for i in range(u.shape[1]):
  a,b=u[:,i],w[:,i]; D=np.c_[np.ones(len(a)),a,b,a*b,a**3]; c=least_squares(lambda c:D@c-v[:,i],fit(D,v[:,i]),loss='cauchy',f_scale=.002).x; m=abs(D@c-v[:,i])<.002; C.append(fit(D[m],v[m,i]))
 c=np.array(C).T
 def f(z):
  w=np.roll(z,-1);return c[0]+c[1]*z+c[2]*w+c[3]*z*w+c[4]*z**3
 def j(z):
  a=np.diag(c[1]+c[3]*np.roll(z,-1)+3*c[4]*z*z);i=np.arange(len(z));a[i,(i+1)%len(z)]=c[2]+c[3]*z;return a
 return np.array([least_squares(lambda z:f(z)-v,v,jac=j,gtol=1e-11).x for v in x])
def panel(p,key,x,f,bounds,linear=True):
 U,V=p['u'],p[key];D=lambda a,w:np.c_[aff(a) if linear else np.ones(len(a)),f(w*a)];
 def loss(w):
  a=D(U,w); return np.mean((a@fit(a,V)-V)**2)
 o=minimize_scalar(loss,bounds=bounds,method='bounded'); print(key,o.x,o.fun);return D(x,o.x)@fit(D(U,o.x),V)
z=cal(s['panel_a'],x); g=panel(s['panel_b'],'v',z,np.cos,(2,3),False)
def cs(a):
 return [panel(s['panel_c'],'v1',a,np.tanh,(1.5,3)),panel(s['panel_c'],'v2',a,lambda x:1/(1+x*x),(.2,.8))]
hz,kz=cs(z); hg,kg=cs(g)
folds=[np.arange(52,72),*np.array_split(np.random.default_rng(9).permutation(len(z)),4)]
def scores(D):
 E=[]
 for v in folds:
  t=np.setdiff1d(np.arange(len(D)),v); r=abs(D[v]@fit(D[t],Y[t])-Y[v]);E.append([r[:,0].mean(),r[:,1:].mean()]);
 return np.round(np.array(E)[[0, np.argmax(np.max(E,axis=1))]],4).tolist()
def q(a):return np.column_stack([a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(a.shape[1]),2)])
from scipy.linalg import qr
from itertools import combinations
F={}
for k,a in zip(['z','g','hz','kz','hg','kg'],[z,g,hz,kz,hg,kg]):
 F[k]=a; F[k+'quad']=q(a); F[k+'tanh']=np.tanh(s['telemetry_vec'][-1]*a);F[k+'cos']=np.cos(1.25*a)
for (k,a),(l,b) in combinations(list(F.items())[:0]+list(zip(['z','g','hz','kz','hg','kg'],[z,g,hz,kz,hg,kg])),2):
 for j in range(4):
  B=np.roll(b,j,axis=1); F[k+l+str(j)]=a*B
  for t in [0.22,1.25]:F[k+l+str(j)+'r'+str(t)]=a/(1+t*B**2)
for j in range(4):
 for k,a in zip(['z','g','hz','kz','hg','kg'],[z,g,hz,kz,hg,kg]):
  F['zz'+k+str(j)]=z*np.roll(z,1,axis=1)*np.roll(a,j,axis=1)
for cols in [slice(0,1),slice(1,None)]:
 for seed in [['g','z'],['g'],['g','z','hz','kz'],['g','z','hg','kg'],[]]:
  v=folds[0]; t=np.setdiff1d(np.arange(len(z)),v); tar=Y[:,cols]; D=np.ones((len(z),1));
  def add(keys):return np.column_stack([D]+[F[k] for k in keys]);
  def loss(A):
   Q=np.linalg.svd(A[t],full_matrices=False); U=Q[0][:,Q[1]>1e-8*Q[1][0]]; h=(U**2).sum(1);r=(tar[t]-U@(U.T@tar[t]))/np.maximum(1-h,.01)[:,None];return (r*r).mean()
  D=add(seed); avail=set(F)-set(seed); best=[]
  for j in range(5):
   k=min(sorted(avail),key=lambda k:loss(add([k])));D=add([k]);avail.remove(k); best.append((k,round(loss(D)**.5,4),round(abs(D[v]@fit(D[t],tar[t])-tar[v]).mean(),4)))
  print(cols,seed,best)
