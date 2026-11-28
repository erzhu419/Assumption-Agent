from operators import *
from itertools import combinations
from scipy.optimize import minimize_scalar

def fit_op(p,key,f,lo,hi):
 u,v=p['u'],p[key]
 def loss(k):
  a=f(u,k); return np.mean((a@fit(a,v)-v)**2)
 k=minimize_scalar(loss,bounds=(lo,hi),method='bounded').x
 a=f(u,k); b=fit(a,v); h=np.sum(a*np.linalg.pinv(a).T,axis=1)
 return (lambda z:f(z,k)@b),k,np.sqrt(loss(k)),np.sqrt(np.mean(((a@b-v)/(1-h[:,None]))**2))
fs={'tanh':(lambda z,k:np.c_[aff(z),np.tanh(k*z)],.3,3.5),'sin':(lambda z,k:np.c_[aff(z),np.sin(k*z)],.3,4.5),'tanh2':(lambda z,k:np.c_[aff(z),np.tanh(k*z),np.tanh(k*z)**2],.3,3.5),'rat':(lambda z,k:np.c_[aff(z),z*z/(1+k*z*z)],.05,1.8),'ratodd':(lambda z,k:np.c_[aff(z),z/(1+k*z*z)],.05,4)}
P={}
for name,p,key in [('B','b','v'),('C','c','v1'),('D','c','v2')]:
 for f,args in fs.items():
  op,*m=fit_op(s['panel_'+p],key,*args); P[name+f]=op
  print(name,f,np.round(m,7))
tr=s['train']; x=coords(tr['x']); Y=np.c_[tr['y'],tr['traj']]; idx=np.arange(len(x)); rng=np.random.default_rng(801)
folds=[(idx[:52],idx[52:]),(idx[:56],idx[56:])]
for group in np.array_split(rng.permutation(idx),4):folds.append((np.setdiff1d(idx,group),group))
large=np.argsort(np.linalg.norm(x,axis=1))[-16:]; folds.append((np.setdiff1d(idx,large),large))

def score(a):
 errors=[]
 for it,iv in folds:
  e=np.abs(a[iv]@fit(a[it],Y[it])-Y[iv]); errors.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(errors)

def show(a,n):
 e=score(a); out.append((n,a.shape[1],*e[0],*e[1],*e[2:6].mean(0),*e[2:6].max(0),*e[-1]))

out=[]
show(pol(x,3),'raw cubic'); show(np.c_[aff(x),*[np.prod(x[:,j],axis=1) for j in combinations(range(4),3)],x**3,*[x[:,i]**2*x[:,j] for i in range(4) for j in range(4) if i!=j]],'odd cubic'); show(np.c_[aff(x),x**3],'diag cubic')
for name,args in fs.items():
 for k in ([.8,1.2,1.5,1.8,2.2,2.6] if name in ('tanh','sin') else [1.5]): show(args[0](x,k),f'raw {name} {k}')
for key,p in P.items():
 z=p(x); show(aff(z),key); show(np.c_[aff(x),z],f'x+{key}'); show(pol(z,2),f'quad {key}'); show(np.c_[aff(z),z**3],f'cubic diag {key}')
for bk in ('Btanh','Btanh2','Bsin'):
 for ck in ('Ctanh','Ctanh2','Cratodd','Drat'):
  b,c=P[bk](x),P[ck](x)
  for name,z,w in [(bk+'>'+ck,b,P[ck](b)),(ck+'>'+bk,c,P[bk](c)),(bk+'|'+ck,b,c)]:
   show(aff(w),name); show(np.c_[aff(z),w],name+'+prev'); show(np.c_[aff(z),w,z*w],name+'+prod'); show(np.c_[aff(x),w],name+'+x')
for name,b,c in [('BCD',P['Btanh'](x),P['Ctanh'](x)),('CD',P['Ctanh'](x),P['Drat'](x)),('C(B)D(B)',P['Ctanh'](P['Btanh'](x)),P['Drat'](P['Btanh'](x)))]:
 for feat,label in [(np.c_[aff(b),c],'join'),(np.c_[aff(b),c,b*c],'join-prod'),(np.c_[aff(x),b,c],'join+x'),(np.c_[aff(b),c,P['Drat'](x)],'join+D')]:show(feat,name+' '+label)
print('model, width, public y/tr, 56/16 y/tr, random mean y/tr, random worst y/tr, radial y/tr');
for r in sorted(out,key=lambda r:max(r[8:10]))[:24]:print(r[:2],np.round(r[2:],5))
for r in sorted(out,key=lambda r:max(r[2:4]))[:14]:print('pub',r[:2],np.round(r[2:],5))
