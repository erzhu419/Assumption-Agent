import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from probe import *
from itertools import product
pairs=[i for k in range(3) for i in combinations_with_replacement(range(4),k)]
D={}
for name,p in zip('BC',[b,c]):
 for key,values in p.items():
  if key=='u':continue
  for d in (1,2):
   h=poly(p['u'],d); coef=ls(h,values);D[name+key+str(d)]=coef
   if d==2:print(name,key,'quadratic coefficients',list(zip(pairs,coef)),sep='\n')
def apply(x,m):return poly(x,1 if len(D[m])==5 else 2)@D[m]
def check(f):
 h=aff(f);g=np.c_[h,np.sum(f[:,:2]*f[:,2:],axis=1)];return [np.mean(np.abs(res(A,Y)/np.maximum(1-np.diag(A@np.linalg.pinv(A)),1e-10).reshape((-1,)+(1,)*(Y.ndim-1)))) for A,Y in [(g,T['y']),(h,T['traj'])]]
L=[]
X=(T['x']-o)/g
for zname,z in [('raw',T['x']),('cal',X)]:
 L.append((check(z),(zname,)))
 for n in range(1,4):
  for path in product(D,repeat=n):
   f=z
   for m in path:f=apply(f,m)
   L.append((check(f),(zname,)+path))
print('ranked paths'); print(*sorted(L,key=lambda r:max(r[0]))[:22],sep='\n')
