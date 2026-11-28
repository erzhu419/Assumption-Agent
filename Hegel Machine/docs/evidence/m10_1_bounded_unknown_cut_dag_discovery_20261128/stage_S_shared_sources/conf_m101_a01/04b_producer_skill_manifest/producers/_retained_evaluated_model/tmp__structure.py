from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.optimize import least_squares
fns={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'softsign':lambda q:q/np.sqrt(1+q*q),'exp':lambda q:np.exp(np.clip(q,-15,15))}
p=s['panel_c']; u=p['u']; h=aff(u); folds_c=np.array_split(np.random.default_rng(34).permutation(len(u)),4)
def bases(h,v,it,key,fn):
 a=h[it]; b=v[it]; b0=ls(a,b); n=len(a); q=h[:,1:]; Q=np.c_[h,q*q];
 if key=='diag+sin':
  def pred(w):return Q@w[:9]+w[9]*np.sin(h@w[10:]);
  w=np.r_[ls(Q[it],b),.15,[0.,.5,.5,.5,.5]]
 elif key=='separable':
  def pred(w):return Q@w[:9]+(np.sin(q*w[9:13])*w[13:17]).sum(1)
  w=np.r_[ls(Q[it],b),[1.]*4,[.15]*4]
 elif key=='aff+link':
  def pred(w):return h@w[:5]+w[5]*fn(h@w[6:]);
  w=np.r_[b0*.2,1,b0*.8]
 else:
  def pred(w):return w[0]+w[1]*fn(h@w[2:]);
  w=np.r_[0.,1.,b0]
 fit=least_squares(lambda w:pred(w)[it]-b,w,max_nfev=220);return fit.x,pred(fit.x),float(np.sqrt(np.mean(fit.fun**2)))
for k in ('v1','v2'):
 v=p[k]; print(k)
 for key,fn in list(fns.items())+[(mode+'_'+name,g) for mode in ('aff+link',) for name,g in fns.items() if name!='exp']+[(g,None) for g in ('diag+sin','separable')]:
  mode=key.split('_')[0]; fits=[bases(h,v[:,i],np.arange(24),mode,fn) for i in range(4)]; errs=[]
  for test in folds_c:
   it=np.setdiff1d(np.arange(24),test); P=np.column_stack([bases(h,v[:,i],it,mode,fn)[1] for i in range(4)]);errs.append(np.mean((P[test]-v[test])**2)**.5)
  print(key,'fit',np.round([f[2] for f in fits],6),'cv',np.round(errs,5));
  if max(f[2] for f in fits)<.006:print('params',np.round([f[0] for f in fits],4))
U,sv,V=np.linalg.svd(j,full_matrices=False); Y=np.c_[y,j@V[:2].T];
for name,z in Z.items():
 F=poly(z,3); b=ls(F,Y); print(name,'linear',np.round(b[1:5].T,4),'quadratic',np.round(b[5:15].T,4),'cubic',np.round(b[15:].T,4));
print('trajectory time fits');
for forms in [('log','t'),('exp','t'),('log','exp'),('exp','exp'),('log','log')]:
 def htime(p): return np.column_stack([{'log':lambda:np.log1p(tt*t),'exp':lambda:-np.expm1(-tt*t),'t':lambda:t}[f]() for tt,f in zip(p,forms)])
 def res(p):
  H=htime(p);return (H@ls(H,j.T)-j.T).ravel()
 opt=least_squares(res,[1.,2.],bounds=([.01,.01],[25,25]));print(forms,opt.x,np.sqrt(np.mean(opt.fun**2)))
