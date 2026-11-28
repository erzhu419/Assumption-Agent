exec(open('/tmp/chains06.py').read().split("trans={'B'")[0])
from itertools import combinations
n=len(y); splits=[(np.arange(52),np.arange(52,n)),(np.arange(20,n),np.arange(20))]
A=design(B,1); Q=design(B,2); O={'q':q,'B':B,'BC':C1(B),'BD':C2(B),'C':C1(q),'D':C2(q)}
for u in ('BC','BD','C','D'):
 for v in ('B','q'):
  for a in (1,-1,.25,-.25): O[f'{v}+{a}{u}']=O[v]+a*O[u]
Y=np.c_[y,tr]; scores=[]
def test(f, idx):
 result=[]
 for i,j in splits:
  sc=np.linalg.norm(f[i],axis=0).clip(1e-10); a=f/sc
  p=a[j]@np.linalg.solve(a[i].T@a[i]+idx*np.eye(a.shape[1]),a[i].T@Y[i]); r=np.mean(abs(p-Y[j]),axis=0);result.append([r[0],np.mean(r[1:])])
 return np.max(result,axis=0)
def push(name,f):
 for ridge in (1e-7,.0005,.005,.025): scores.append((test(f,ridge),name,ridge))
for k,h in O.items():
 push((k,'quad'),design(h,2))
 for d in (1,2):
  push((k,'augment',d),np.c_[Q,design(h,d)[:,1:]])
 for i in range(4):
  for s in (.8,2,4,8,16):
   for p in (np.tanh,np.sin,lambda v:1/(1+v*v)):
    g=p(s*h[:,i:i+1]);push((k,i,s,p.__name__),np.c_[Q,A*g]);
 for p in (np.tanh,np.sin,lambda v:1/(1+v*v),lambda v:v**3,lambda v:abs(v)):
  for s in (1,3,8):
   g=p(s*h);push((k,s,p.__name__,'elem'),np.c_[Q,g]);push((k,s,p.__name__,'elemprod'),np.c_[Q,g,*[h[:,i:i+1]*g for i in range(4)]]);
for ii in (0,1): print('TOP',ii,sorted(scores,key=lambda s:s[0][ii])[:14]);
# Fit gates with the quadratic base on B and the score in a separate coordinate system.
from scipy.optimize import least_squares
_,_,vv=np.linalg.svd(tr,full_matrices=False)
for label,target in [('y',y),('t',tr@vv[0])]:
 out=[]
 for name in ('q','B','BC','BD','C','D'):
  h=O[name]; H=design(h,1); sols=[]
  for i,j in splits:
   def ff(p):return np.c_[Q,A*np.tanh(H@p)[:,None]]
   def res(p):
    f=ff(p)[i];return f@ls(f,target[i])-target[i]
   pp=np.vstack([np.eye(5)[1:]*s for s in (2,7,14)] + [np.random.default_rng(901).normal(0,3,(10,5))]); o=min((least_squares(res,p,bounds=(-25,25),max_nfev=95) for p in pp),key=lambda r:r.cost); f=ff(o.x); sols.append((np.mean(abs(f[j]@ls(f[i],target[i])-target[j])),np.sqrt(np.mean(o.fun**2)),o.x.round(3)))
  print('GATE',label,name,sols,flush=True)
