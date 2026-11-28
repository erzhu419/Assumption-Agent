exec(open('/tmp/scan.py').read().split("p=s['panel_a'];")[0])
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
p=s['panel_a']; U,V=p['u'],p['v']; a=design(U); r=np.linalg.norm(a@ls(a,V)-V,axis=1); use=np.argsort(r)[:45]; U,V=U[use],V[use]
def mons(d):return [i for k in range(1,d+1) for i in itertools.combinations_with_replacement(range(4),k)]
def poly(u,ind):return np.column_stack([np.ones(len(u))]+[np.prod(u[:,i],axis=1) for i in ind])
def cv(a,v):
 e=np.empty_like(v)
 for i,j in KFold(5,shuffle=True,random_state=42).split(a): e[j]=a[j]@ls(a[i],v[i])-v[j]
 return np.sqrt(np.mean(e**2,axis=0))
for u,v in [(U,V),(V,U)]:
 for d in range(1,5):
  a=poly(u,mons(d)); co=ls(a,v); print('poly',d,'fit',np.sqrt(np.mean((a@co-v)**2,axis=0)),'cv',cv(a,v));
  if d==2: print(list(zip(mons(d),np.round(co[1:],4))))
 z=np.roll(u,-1,axis=1); print('adjacent empirical')
 for j in range(4):
  rr=v[:,j]-design(u)@ls(design(u),v[:,j]); out=[]
  for i,k in itertools.combinations_with_replacement(range(4),2):
   for nm,w in [('product',u[:,i]*u[:,k]),('sqsum',(u[:,i]+u[:,k])**2),('sinprod',np.sin(u[:,i]*u[:,k])),('tanhprod',np.tanh(u[:,i]*u[:,k])),('absdiff',abs(u[:,i]-u[:,k])),('absprod',abs(u[:,i]*u[:,k])),('tanhdiff',np.tanh(2*(u[:,i]-u[:,k]))),('signedsq',np.sign(u[:,i]-u[:,k])*(u[:,i]-u[:,k])**2),('sindiff',np.sin(2*(u[:,i]-u[:,k])))]:
    out.append((float(cv(design(u,w),v[:,j])),nm,i,k))
  print(j,sorted(out)[:8])
print('archive context');
