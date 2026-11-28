from infer import *
from scipy.optimize import root
import itertools as it
p=s['panel_a']; u,v=p['u'],p['v']; ai=np.where(np.linalg.norm(v-u,axis=1)<4)[0]
affs=lambda x:np.c_[np.ones(len(x)),x]
B=ls(affs(s['panel_b']['u']),s['panel_b']['v']); b=lambda x:affs(x)@B
p=s['panel_c']; C=[panel(p['u'],p[f'v{i}'],fun[f],True,np.arange(len(p['u'])))[1] for i,f in [(1,'soft'),(2,'sqrat')]]
c,d=C
x=np.r_[s['train']['x'],s['test']['x']]; y=s['train']['y']; z=s['train']['traj']; P=np.linalg.svd(z[:52],full_matrices=False)[2][:2]; Y=np.c_[y,z@P.T];
def maps():
 yield 'raw',x
 for name in ['affine','rat','tanh','sin','root']:
  if name=='affine':
   W=ls(affs(u[ai]),v[ai]); a=lambda x:affs(x)@W
  else:
   k,a,w=panel(u,v,fun[name],True,ai)
  inv=np.array([root(lambda q:a(q[None])[0]-row,row).x for row in x]); print('inversion',name,np.max(abs(a(inv)-x)))
  yield name+'+',a(x); yield name+'-',inv

def gram(p,q):return np.einsum('ni,nj->nij',p,q).reshape(len(p),-1)
def variants(q):
 for bn,h in [('q',q),('bq',b(q)),('biq',np.linalg.solve(B[1:].T,(q-B[0]).T).T)]:
  p,r=c(h),d(h)
  yield bn,h,p,r
  if bn=='q': yield 'bcd',b(q),p,r

def bases(h,p,r):
 cp=gram(p,r); dots=np.sum(p*r,axis=1)
 yield 'lin',affs(np.c_[p,r]); yield 'dot',affs(np.c_[p,r,dots]);yield 'bilin',affs(np.c_[p,r,cp]); yield 'hdot',affs(np.c_[h,p,r,dots]);yield 'hbilin',affs(np.c_[h,p,r,cp]);yield 'ppdd',affs(np.c_[p,r,dots,gram(p,p)[:,[4*i+j for i,j in it.combinations_with_replacement(range(4),2)]],gram(r,r)[:,[4*i+j for i,j in it.combinations_with_replacement(range(4),2)]]]);yield 'h',affs(h);yield 'hpoly2',affs(np.c_[h,gram(h,h)[:,[4*i+j for i,j in it.combinations_with_replacement(range(4),2)]]]);yield 'hb',affs(np.c_[h,p,r,gram(h,p),gram(h,r)])

def score(a,tr,te,reg=0):
 f=a[tr]; X=a[te];
 if reg:
  sc=np.maximum(np.std(f,axis=0),.15); J=np.eye(f.shape[1])*reg;J[0,0]=0;f=f/sc;X=X/sc; w=np.linalg.solve(f.T@f+J,f.T@Y[tr]);
 else:w=ls(f,Y[tr]);
 out=X@w;return np.array([np.mean(abs(out[:,0]-y[te])),np.mean(abs(out[:,1:]@P-z[te]))])
if __name__=='__main__':
 idx=np.arange(len(y));folds=[(idx[:52],idx[52:])];rng=np.random.default_rng(5324)
 for _ in range(8):
  i=rng.permutation(idx);folds.append((i[:52],i[52:]));
 rows=[]
 for an,q in maps():
  for bn,h,p,r in variants(q):
   for fn,a in bases(h,p,r):
    for reg in ([0,.01,.1] if a.shape[1]>18 else [0]):
     cv=np.array([score(a,tr,te,reg) for tr,te in folds]); rows.append((an,bn,fn,reg,*cv[0],*cv.mean(0),*cv.max(0)))
 for j in [4,5,6,7]:
  print('sorting',j)
  for row in sorted(rows,key=lambda row:row[j])[:12]:print(*row[:4],np.round(row[4:],6))
