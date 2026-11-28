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
from itertools import combinations, product
from scipy.linalg import qr
B=aff(g); y=Y[:,0]; ids=np.arange(len(y)); extra={}
# The cosine features before the output mixing are a distinct coordinate system.
u=np.cos(2.6*z); pp=np.c_[u,g];
for k,a in [('u',u),('g',g),('h',hg),('k',kg),('hz',hz),('kz',kz),('z',z)]:
 for i in range(4):
  v=a[:,i]
  for w in [.22, .6, 1.25, 2.15, 2.6, 3.5]:
   for nm,f in [('C',np.cos),('S',np.sin),('T',np.tanh),('R',lambda v:1/(1+v*v))]:extra[(k,i,nm,w)]=f(w*v)
  for j in range(i,4):extra[(k,i,j,'prod')]=v*a[:,j]
  for j in range(4):
   for nam,b in [('u',u),('g',g)]:
    p=b[:,j];extra[(k,i,nam,j,'prod')]=v*p
    for w in [.22, 1.25,2.15]:
     extra[(k,i,nam,j,'quot',w)]=p/(1+w*v*v)
     for t in [1,-1]:
      extra[(k,i,nam,j,'dot',t,w)]=np.cos(w*(p+t*v))
      extra[(k,i,nam,j,'tanh',t,w)]=np.tanh(w*(p+t*v))
for k,a in [('g',g),('u',u)]:
 for i,j,l in combinations_with_replacement(range(4),3):extra[(k,i,j,l,'cube')]=a[:,i]*a[:,j]*a[:,l]
E=np.column_stack(list(extra.values())); keys=list(extra)
for base in [B,np.c_[B,q(g)]]:
 out=[]
 for v in folds:
  t=np.setdiff1d(ids,v); U=qr(base[t],mode='economic')[0]; r=y[t]-U@(U.T@y[t]); R=E[t]-U@(U.T@E[t]); norms=np.linalg.norm(R,axis=0); good=norms>1e-7; R=R/np.maximum(norms,1e-7); R[:,~good]=0
  print('singles',[(keys[j],round(abs(r@R[:,j])/np.linalg.norm(r),4)) for j in np.argsort(abs(r@R))[-6:][::-1]])
  cand=[]
  for j in range(10):
   l=np.argmax(abs(r@R));cand.append(l); D=np.c_[base,E[:,cand]]; U=qr(D[t],mode='economic')[0]; r=y[t]-U@(U.T@y[t]); print('step',j,keys[l],round(np.sqrt(np.mean(r*r)),4),round(abs(D[v]@fit(D[t],y[t])-y[v]).mean(),4),end='; ')
  print()
