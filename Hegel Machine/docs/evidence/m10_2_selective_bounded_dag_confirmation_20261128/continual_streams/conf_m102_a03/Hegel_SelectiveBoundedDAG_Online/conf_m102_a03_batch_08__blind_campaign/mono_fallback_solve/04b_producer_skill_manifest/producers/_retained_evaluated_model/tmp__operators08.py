import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; D=np.load('/tmp/feat08.npz'); z,y,t=D['z'],D['y'],D['t']; n=52
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
poly=lambda x,d:np.column_stack([np.prod(x[:,i],axis=1) for i in combinations_with_replacement(range(x.shape[1]),d)])
ops={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'bump':lambda x:1/(1+x*x),'exp':lambda x:np.exp(-x*x),'quad':lambda x:x*x,'cube':lambda x:x**3}
enc={'z':z}
for pn in ['panel_b','panel_c']:
 p=s[pn];u=p['u'];
 print(pn, 'range',u.min(),u.max())
 for key,v in p.items():
  if key=='u':continue
  models=[]
  for bn,base in [('aff',aff),('one',lambda x:aff(x)[:,:1]),('affquad',lambda x:np.c_[aff(x),poly(x,2)])]:
   for f,fn in ops.items():
    def basis(x,h):return np.c_[base(x),fn(h*x)]
    def loss(h):
     A=basis(u,h);return np.mean((A@fit(A,v)-v)**2)**.5
    grid=np.linspace(.3,8,100); i=np.argmin([loss(h) for h in grid]); h=minimize_scalar(loss,bounds=(grid[max(0,i-1)],grid[min(99,i+1)]),method='bounded').x
    models.append((loss(h),bn,f,h,basis(z,h)@fit(basis(u,h),v),fit(basis(u,h),v)))
  models.sort(key=lambda m:m[:4]);
  for m in models[:9]:print(key,m[:4]);
  print('coef',models[0][-1].round(5)); enc[pn+key]=models[0][-2]
  for f in ['sin','tanh','rat']:
   m=next(m for m in models if m[1]=='aff' and m[2]==f);enc[pn+key+f]=m[-2]
np.savez('/tmp/enc08.npz',**enc)
for name,b in enc.items():
 for typ,f in [('lin',lambda v:aff(v)),('quad',lambda v:np.c_[aff(v),poly(v,2)]),('cross',lambda v:np.c_[aff(v),poly(v,2)[:,[1,2,3,5,6,8]]])]:
  A=f(b); Y=A@fit(A[:n],y[:n]); T=A@fit(A[:n],t[:n]); print(name,typ,*[round(float(v),5) for v in (np.mean(abs(Y[n:]-y[n:])),np.mean(abs(Y[:n]-y[:n])),np.mean(abs(T[n:]-t[n:])))])
