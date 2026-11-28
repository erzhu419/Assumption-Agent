exec(open('/tmp/fit07.py').read().split('for name,b')[0])
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=5,suppress=True,linewidth=150)
W=w[:n]
def f1(z):return np.c_[np.ones(len(z)),z,pair(z,False)]
q1=reg(f1(u),p['v1'],0); print('v1 coefficients',q1, 'RMSE',np.std(f1(u)@q1-p['v1']))
mods=[]
for k in fun:
 if k=='lin':continue
 for tail in ['sq','cross','all','none']:
  def A(z,g,k=k,tail=tail):
   z2=fun[k](g*z); a=np.c_[np.ones(len(z)),z,z2]; return np.c_[a, {'sq':z*z,'cross':pair(z,False),'all':pair(z),'none':np.empty((len(z),0))}[tail]]
  def fit(train, val):
   b=p['v2']; fn=lambda g:np.mean((A(u[train],g)@reg(A(u[train],g),b[train],0)-b[train])**2)
   grid=np.linspace(.16,4,40); j=np.argmin([fn(g) for g in grid]); g=minimize_scalar(fn,bounds=(grid[max(0,j-1)],grid[min(j+1,len(grid)-1)]),method='bounded').x
   q=reg(A(u[train],g),b[train],0);return np.mean(abs(A(u[val],g)@q-b[val])),g,q
  ac=[fit(np.setdiff1d(np.arange(24),v),v)[0] for v in np.array_split(np.arange(24),4)]
  e,g,q=fit(np.arange(24),np.arange(24)); mods.append((np.mean(ac),e,k,tail,g,q,A))
for a in sorted(mods,key=lambda a:a[0])[:10]:print('v2',a[:5])
_,_,_,_,g,q,A=min(mods,key=lambda a:a[0]); print('v2 coefficients',q)
V1=f1(W)@q1; V2=A(W,g)@q
for b,name in [(y,'scalar'),(tr,'traj')]:
 res=[]
 for key,z in [('w',W),('v1',V1),('v2',V2),('v1v2',np.c_[V1,V2]),('wv1',np.c_[W,V1]),('wv2',np.c_[W,V2]),('all',np.c_[W,V1,V2])]:
  Fs=[('linear',z),('diag2',np.c_[z,z*z]),('local2',np.c_[z,z*z,z*np.roll(z,-1,axis=1)]),('poly2',poly(z,2)),('tanh',np.c_[z,np.tanh(z)]),('sin',np.c_[z,np.sin(z)]),('atan',np.c_[z,np.arctan(z)])]
  for ft,F in Fs:
   F=np.c_[np.ones(n),F]
   for r in [0,.02,.2,1,3]:
    e=[np.mean(abs(F[v]@reg(F[a],b[a],r)-b[v])) for a,v in folds]; res.append((np.mean(e),max(e),e[0],key,ft,r))
 print(name, 'best', sorted(res)[:15]); print(name,'linear', sorted(a for a in res if a[4]=='linear' and a[5]==0))
