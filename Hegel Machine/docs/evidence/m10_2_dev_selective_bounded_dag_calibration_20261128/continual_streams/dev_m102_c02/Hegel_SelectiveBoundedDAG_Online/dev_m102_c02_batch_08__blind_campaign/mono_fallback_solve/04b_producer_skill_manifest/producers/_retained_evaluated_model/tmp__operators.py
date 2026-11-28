import pickle,numpy as np
from itertools import combinations_with_replacement as cr
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=130)
def lin(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return lstsq(x,y,cond=1e-10)[0]
def loo(x,y):
 e=y-x@ls(x,y); h=(np.linalg.qr(x)[0]**2).sum(1);return np.mean(abs(e/(1-h[:,None])))
def powers(x,d):return np.column_stack([np.prod(x[:,i],1) for i in cr(range(4),d)])
def transforms(x):
 return {'sq':x*x,'cube':x**3,'sqcube':np.c_[x*x,x**3],'abs':abs(x),'pos':np.maximum(x,0),'tanh':np.tanh(x),'sin':np.sin(x),'sin2':np.sin(2*x),'sin3':np.sin(3*x),'cos':np.cos(x),'cos2':np.cos(2*x),'cos3':np.cos(3*x),'rat':x/(1+x*x),'rat2':x/(1+2*x*x),'ssq':x*x/(1+x*x),'atan':np.arctan(x),'quad':powers(x,2),'cyc':x*np.roll(x,1,axis=1),'sumabs':abs(x+np.roll(x,1,axis=1)),'difabs':abs(x-np.roll(x,1,axis=1))}
def fit_panel(name):
 p=s['panel_'+name]; u=p['u']; out={}
 for k,v in p.items():
  if k=='u':continue
  x,y=u,v
  if name=='a':
   a=lin(x); r=np.linalg.norm(y-a@ls(a,y),axis=1); keep=r<4*np.median(r); x,y=x[keep],y[keep]
  tr=transforms(x); LL=[]
  for f,z in list(tr.items())+[('lin',np.empty((len(x),0)) )]:
   a=np.c_[lin(x),z]; LL.append((loo(a,y),f,ls(a,y)))
  print(name,k,'best',[(round(e,5),f) for e,f,b in sorted(LL)[:13]])
  for e,f,b in sorted(LL)[:3]: print(f,b)
  if name=='a':
   a=lin(x); r=y-a@ls(a,y);print('u|residual sample',np.c_[x,r][:15]); print('train/test bounds relative to panel', x.min(0),x.max(0),y.min(0),y.max(0))
  out[k]=(x,y,sorted(LL))
 return out
pa,pb,pc=map(fit_panel,'abc')
X=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; u,v,rr=pa['v']; a=ls(lin(u),v); C=(X-a[0])@np.linalg.inv(a[1:]);
from scipy.optimize import root
cal={'raw':X,'aff':C};
for e,f,b in rr[:4]:
 if f=='lin':continue
 def fn(z):
  z=z.reshape(-1,4);return (np.c_[lin(z),transforms(z)[f]]@b-X).ravel()
 sol=root(fn,C.ravel()); print('inverse',f,sol.success,np.mean(abs(fn(sol.x))));
 if sol.success:cal['inverse_'+f]=sol.x.reshape(-1,4)
for ca,z in cal.items():
 cand=[]
 for nf,q in [('x',z)]+list(transforms(z).items()):
  for panel,ps in [('b',pb),('c',pc)]:
   for k,(u,v,rr) in ps.items():
    for e,f,b in rr[:3]:
     if q.shape[1]!=4:continue
     d=lin(q) if f=='lin' else np.c_[lin(q),transforms(q)[f]]; w=d@b
     for g,F in [('w',lin(w)),('xw',np.c_[lin(z),w]),('w2',np.c_[lin(w),powers(w,2)]),('xwprod',np.c_[lin(z),w,z*w]),('w_tanh',np.c_[lin(w),np.tanh(w)])]:
      P=F@ls(F[:52],Y[:52]); M=np.mean(abs(P[52:]-Y[52:]),0);cand.append((M[0],M[1:].mean(),nf,panel,k,f,g))
 print('CAND',ca,'by sum',sorted(cand,key=lambda r:sum(r[:2]))[:8],'by traj',sorted(cand,key=lambda r:r[1])[:5])
