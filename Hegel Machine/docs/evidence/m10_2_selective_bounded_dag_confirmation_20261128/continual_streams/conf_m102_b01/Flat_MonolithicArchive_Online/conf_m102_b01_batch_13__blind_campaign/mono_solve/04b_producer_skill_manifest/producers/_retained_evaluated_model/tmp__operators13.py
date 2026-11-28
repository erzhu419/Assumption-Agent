import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda u:np.c_[np.ones(len(u)),u]
quad=lambda u:np.c_[[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]].T
fn={'lin':lambda x,k:x,'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'tanh':lambda x,k:np.tanh(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'rat':lambda x,k:x/(1+k*x*x),'sq':lambda x,k:x*x}
def scan(p,key,kinds):
 u,v=p['u'],p[key]; records=[]
 for names,feat,ini in kinds:
  def res(k):
   d=aff(feat(u,k));return (d@ls(d,v)-v).ravel()
  o=least_squares(res,ini,bounds=(.06,6),max_nfev=100); d=aff(feat(u,o.x)); w=ls(d,v); h=np.sum(d*np.linalg.pinv(d).T,1); e=res(o.x).reshape(v.shape); records.append((np.sqrt(np.mean((e/(1-h[:,None]))**2)),np.std(e),names,o.x,w,feat))
 for loo,err,n,k,w,_ in sorted(records,key=lambda a:a[0])[:9]:print(key,n,np.round(k,4),round(err,6),round(loo,6), 'coef',np.round(w,3).tolist() if err<.004 else '')
 return sorted(records,key=lambda a:a[0])
B=[]; C=[]
for base in ('cos','tanh','asinh','rat','sin'):
 for extra in ('none','lin','sin','cos','sq','tanh','asinh','rat','quad','sinprod'):
  def f(u,k,base=base,extra=extra):
   a=fn[base](u,k[0]);
   if extra=='none':return a
   if extra=='quad':return np.c_[a,quad(u)]
   if extra=='sinprod':return np.c_[a,np.sin(k[1]*u*np.roll(u,1,axis=1))]
   return np.c_[a,fn[extra](u,k[1])]
  (B if base=='cos' else C).append((base+'+'+extra,f,[2,1.5] if extra not in ('none','quad') else [2]))
for rot in [-1,1,2,0]:
 def f(u,k,rot=rot):return np.tanh(k[0]*u)+k[2]*np.sin(k[1]*np.roll(u,rot,axis=1))
 C.append((f'coupled{rot}',f,[1.75,2.1,.3]));
for rot in [-1,1,2,0]:
 def f(u,k,rot=rot):return np.cos(k[0]*u)+k[2]*np.sin(k[1]*np.roll(u,rot,axis=1))
 B.append((f'coupled{rot}',f,[2,2,.15]));
b=scan(s['panel_b'],'v',B); c=scan(s['panel_c'],'v1',C)
p=s['panel_c'];u,v=p['u'],p['v2']; print('C2 quad',np.round(ls(aff(np.c_[u,quad(u)]),v),4)); print('C2 cross only',np.std(aff(np.c_[u,u*np.roll(u,1,axis=1)])@ls(aff(np.c_[u,u*np.roll(u,1,axis=1)]),v)-v));print('telemetry',s['telemetry_vec'],'times',s['eval_times'])
# Compare candidate stage inputs and outputs, with fixed basis and fit on the first 52 rows.
a=s['panel_a']; co=[]
for j in range(4):
 d=aff(a['u'][:,j]); z=a['v'][:,j]; ix=np.arange(len(z));
 for _ in range(10):ix=np.argsort(abs(d@ls(d[ix],z[ix])-z))[:40]
 e=abs(d@ls(d[ix],z[ix])-z);ix=e<.001;co.append(ls(d[ix],z[ix]))
co=np.array(co);x=(s['train']['x']-co[:,0])/co[:,1];T=s['train']['traj'];Y=np.c_[s['train']['y'],T]; n=52
ops=[]
for key,records in [('b',b),('c1',c)]:
 for _,_,name,k,w,f in records[:4]:ops.append((key+name,lambda u,k=k,w=w,f=f:aff(f(u,k))@w))
w=ls(aff(np.c_[u,quad(u)]),v);ops.append(('c2',lambda u:aff(np.c_[u,quad(u)])@w));
coords=[('z',x)]+[(name,f(x)) for name,f in ops]; coords += [(bn+':'+cn,cf(bf(x))) for bn,bf in ops if bn.startswith('b') for cn,cf in ops if cn.startswith('c')]; coords += [(bn+':'+cn,cf(bf(x))) for bn,bf in ops if bn.startswith('c') for cn,cf in ops if cn.startswith('b')];
D=[]
for name,z in coords:
 for k,d in [('lin',aff(z)),('quad',aff(np.c_[z,quad(z)]))]:
  e=np.mean(abs(d[n:]@ls(d[:n],Y[:n])-Y[n:]),0);D.append((e[1:].mean(),e[0],name,k))
print('trajectory',sorted(D)[:14]);print('scalar',sorted(D,key=lambda a:a[1])[:14]);
