import pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x): return np.c_[np.ones(len(x)),x]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
a,b,c=[s['panel_'+k] for k in 'abc']; U=aff(a['u']); V=a['v']; m=np.arange(len(U))
for _ in range(8):
 A=fit(U[m],V[m]); r=np.linalg.norm(U@A-V,axis=1); m=np.argsort(r)[:38]
print('A',A,'resids',np.sort(r)); A=fit(U[r<1.],V[r<1.]); print('A refined',A)
B=fit(aff(b['u']),b['v']); u=c['u'];
def rm(g,v):
 rr=v-g@fit(g,v); return np.sqrt(np.mean(rr**2))
F={'atan':lambda x,k:np.arctan(k*x), 'tanh':lambda x,k:np.tanh(k*x),'soft':lambda x,k:x/np.sqrt(1+k*x*x),'rat':lambda x,k:x/(1+k*x*x),'sin':lambda x,k:np.sin(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'sat2':lambda x,k:x*x/(1+k*x*x),'cos':lambda x,k:np.cos(k*x),'log':lambda x,k:np.log1p(k*x*x),'exp2':lambda x,k:np.exp(-k*x*x)}
C=[]
for label in ('v1','v2'):
 v=c[label]; rs=[]
 for name,f in F.items():
  for extra in (False,True):
   features=lambda q,k: np.c_[aff(q),f(q,k)] if extra else aff(f(q,k)); opt=minimize_scalar(lambda k:rm(features(u,k),v),bounds=(.005,4),method='bounded'); rs.append((opt.fun,name,extra,opt.x))
 print(label,'best direct',sorted(rs)[:10]);
 for name,f in {'atan':lambda x,k:np.tan(k*x),'tanh':lambda x,k:np.arctanh(k*x),'soft':lambda x,k:x/np.sqrt(1-k*x*x),'sin':lambda x,k:np.arcsin(k*x)}.items():
  opt=minimize_scalar(lambda k:rm(aff(u),f(v,k))/k,bounds=(.001,.999/np.max(abs(v))),method='bounded'); print('inverse',name,opt.fun,opt.x)
 if label=='v1': C.append(lambda q:aff(np.arctan(2.15*q))@fit(aff(np.arctan(2.15*u)),c['v1']))
 else:
  _,name,_,k=next(rr for rr in sorted(rs) if rr[2]); print('best v2 coefficients',fit(np.c_[aff(u),F[name](u,k)],v)); C.append(lambda q:np.c_[aff(q),F[name](q,k)]@fit(np.c_[aff(u),F[name](u,k)],c['v2']))
def coordinates(x):
 ca=(x-A[0])@np.linalg.inv(A[1:]); aa=aff(x)@A
 return {'x':x,'ca':ca,'aa':aa, 'b':aff(x)@B,'ba':aff(ca)@B,'ab':(aff(x)@B-A[0])@np.linalg.inv(A[1:]),'bAA':aff(aa)@B}
def powers(x,d): return np.column_stack([np.prod(x[:,j],axis=1) for j in cr(range(x.shape[1]),d)])
if __name__=='__main__':
 x=s['train']['x']; yz=np.c_[s['train']['y'],s['train']['traj']]; n=52; fs=[]
 base=coordinates(x)
 for j,cmap in enumerate(C):
  for key,q in list(coordinates(x).items()):base[f'c{1+j}:{key}']=cmap(q)
 for key,q in base.items():
  for nm,w in [('id',q)]+[(f'{nm}:{k}',f(q,k)) for nm,f in F.items() if nm in ('atan','tanh','soft','asinh') for k in (.22,.55,1.25,2.15)]:
   for d in (1,2,3):
    design=np.column_stack([np.ones(len(w))]+[powers(w,j) for j in range(1,d+1)]); h=design[:n];
    for lam in (1e-7,.001,.02):
     pred=design@np.linalg.solve(h.T@h+lam*np.eye(h.shape[1]),h.T@yz[:n]); r=abs(yz-pred); fs.append((r[n:,0].mean(),r[n:,1:].mean(),r[:n,0].mean(),r[:n,1:].mean(),key,nm,d,lam))
 for axis in (0,1):
  print('TOP',axis); print(*sorted(fs,key=lambda row:row[axis])[:12],sep='\n')
