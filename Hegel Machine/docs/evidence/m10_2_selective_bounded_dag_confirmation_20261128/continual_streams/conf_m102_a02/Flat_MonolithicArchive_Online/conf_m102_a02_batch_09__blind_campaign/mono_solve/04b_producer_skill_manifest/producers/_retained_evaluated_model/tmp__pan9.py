import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as comb
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
L=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
A=lambda x:np.c_[np.ones(len(x)),x]
Q=lambda x:np.c_[A(x),*[np.prod(x[:,i],axis=1) for i in comb(range(x.shape[1]),2)]]
u,v=(s['panel_a'][k] for k in ['u','v']); pars=[]
def ac(x,j,k):
 a,b=x[:,j],x[:,k]; return np.c_[np.ones(len(x)),a,b,a*b,a**3]
for j in range(4):
 report=[]
 for k in set(range(4))-{j}:
  f=ac(u,j,k); c=least_squares(lambda c:f@c-v[:,j],[0,1,.2,.2,.2],loss='soft_l1',f_scale=.00015).x
  r=abs(f@c-v[:,j]); keep=r<max(.001,5*np.median(r)); c=L(f[keep],v[keep,j]); r=abs(f@c-v[:,j]); report.append((np.median(r),k,c));print('cal',j,k,c,np.quantile(r,[.5,.85,.93,.96,1]))
 pars.append(min(report,key=lambda a:a[0])[1:])
def inverse(x):
 def f(z):
  z=z.reshape(x.shape);return (np.stack([ac(z,j,k)@c for j,(k,c) in enumerate(pars)],axis=1)-x).ravel()
 # Vectorized independent Newton solves.
 z=x.copy()
 for _ in range(25):
  r=f(z).reshape(x.shape); jac=np.zeros((len(x),4,4))
  for j,(k,c) in enumerate(pars):
   jac[:,j,j]=c[1]+c[3]*z[:,k]+3*c[4]*z[:,j]**2; jac[:,j,k]=c[2]+c[3]*z[:,j]
  z-=np.clip(np.linalg.solve(jac,r[...,None])[...,0],-1,1)
 print('inversion max',abs(f(z)).max());return z
x=inverse(s['train']['x']); y=s['train']['y']; traj=s['train']['traj']; target=np.c_[y,traj]; t=s['eval_times']; print('x range',x.min(0),x.max(0))
def score(f,tag):
 p=f@L(f[:52],target[:52]); err=abs(p-target); print(tag,f.shape[1],np.round([err[:52,0].mean(),err[52:,0].mean(),err[:52,1:].mean(),err[52:,1:].mean()],6))
bp=s['panel_b']; B=lambda x:A(x)@L(A(bp['u']),bp['v']); b=B(x)
for f,tag in [(A(x),'lin'),(Q(x),'quad'),(np.c_[Q(x),x**3],'cubic direct'),(np.c_[Q(x),b**3],'cubic B'),(np.c_[Q(x),*[np.prod(x[:,i],axis=1) for i in comb(range(4),3)]],'cubic full'),(np.c_[A(x),np.tanh(b)],'lin+tanhB')]:score(f,tag)
cp=s['panel_c']; u=cp['u']; v1=cp['v1']; v2=cp['v2']; print('C extrema',v1.min(0),v1.max(0),v2.min(0),v2.max(0))
funcs={'tanh':lambda z,p:np.tanh(p*z),'sin':lambda z,p:np.sin(p*z),'cube':lambda z,p:z**3,'square':lambda z,p:z**2,'gauss':lambda z,p:np.exp(-p*z*z),'rat':lambda z,p:z/(1+p*z*z),'soft':lambda z,p:z/np.sqrt(1+p*z*z)}
for label,v in [('v1',v1),('v2',v2)]:
 best=[]
 for name,fun in funcs.items():
  for label0,W,Wx in [('u',u,x),('b',B(u),b)]+([('v1',v1,None)] if label=='v2' else []):
   for base,baseq,bname in [(np.ones((len(W),1)),np.ones((len(x),1)),'const'),(A(W),A(Wx) if Wx is not None else None,'aff'),(Q(W),Q(Wx) if Wx is not None else None,'quad')]:
    for p in [.3,.6,1.,1.5,2.]:
     f=np.c_[base,fun(W,p)]; pinv=np.linalg.pinv(f); rr=f@pinv@v-v; e=np.sqrt(np.mean((rr/np.maximum(1-np.sum(f*pinv.T,1),1e-8)[:,None])**2)); best.append((e,name,label0,bname,p, f, None if Wx is None else np.c_[baseq,fun(Wx,p)]))
 for e,nm,src,bnm,p,f,fq in sorted(best,key=lambda a:a[0])[:8]:
  print(label,e,nm,src,bnm,p,np.std(f@L(f,v)-v));
  if fq is not None: h=fq@L(f,v); score(np.c_[Q(x),h],label+' +q');score(np.c_[A(h),h**2],'h + squares')
