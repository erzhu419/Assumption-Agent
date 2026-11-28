import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from maps14 import *
from sklearn.linear_model import Lasso
rng=np.random.default_rng(100); folds=[(p[20:],p[:20]) for p in [rng.permutation(len(y)) for _ in range(5)]]
def cv(a,t):
 a=np.c_[np.ones(len(a)),a]; return np.mean([np.mean(abs(a[j]@ls(a[i],t[i])-t[j])) for i,j in folds])
def track(tag,a): print(tag,a.shape,round(cv(a,y),5),round(cv(a,z),5))
for shift in range(4):
 c=b*np.roll(q,shift,axis=1); track(('cyc',shift),np.c_[b,c]); track(('cycsum',shift),np.c_[b,c.sum(1)])
inter=(b[:,:,None]*q[:,None,:]).reshape(len(b),16); track('all cross',np.c_[b,inter]); track('all cross+b2',np.c_[poly(b,2)[:,1:],inter]);
for stage in [q,b,fc1(q),fc2(q),fc1(b),fc2(b)]:
 track('stage',np.c_[b,stage]);track('cross stage',np.c_[b,(b[:,:,None]*stage[:,None,:]).reshape(len(b),16)])
res=[]
cs=[b*np.roll(q,i,axis=1) for i in range(4)]
for shift,c in enumerate(cs):
 for name,arr in [('q',q),('b',b),('c1',fc1(q)),('c2',fc2(q))]:
  hs=[(str(j),arr[:,j]) for j in range(4)]+[(f'{j}{k}{sgn}',arr[:,j]+sgn*arr[:,k]) for j in range(4) for k in range(j+1,4) for sgn in [-1,1]]
  for j,h in hs:
   for p in [.22,1.25,2.15,4.]:
    g=1/(1+p*h*h); bg=b*g[:,None]
    for kind,a in enumerate([np.c_[bg,c],np.c_[bg,c,bg*h[:,None]],np.c_[bg,c,bg*h[:,None]**3],np.c_[bg,c,bg*h[:,None],bg*h[:,None]**3],np.c_[bg,b,c],np.c_[bg,c*g[:,None],c],np.c_[bg,c,h,h*h*g,h*g],np.c_[bg,c.sum(1)]]):
     res.append((cv(a,y),shift,name,j,p,kind))
print('best scalar',sorted(res)[:24]);
print('A raw design errors:')
for freq in [2.15,2.4,2.45,2.5]:
 uu=A['u'][good]; vv=A['v'][good]; d=np.c_[poly(uu,1),np.sin(freq*uu),uu**3]; coef=ls(d,vv); print(freq,np.mean(abs(d@coef-vv)),np.round(coef[5:],4))
