import numpy as np,itertools
from scipy.special import erf
D=np.load('/tmp/train14.npz'); z,g,j,k,y=(D[a] for a in ('z','g','j','k','y')); y=y[:len(j)]; z,g,j,k=[a[:len(y)] for a in (z,g,j,k)]; n=len(y)
L=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
A=lambda *a:np.column_stack([np.ones(n),*a]); roll=lambda a,r=1:np.roll(a,r,axis=1)
v=z*roll(g); groups={'z':z,'g':g,'j':j,'k':k,'v':v}; core=A(z,g,v)
for a,na in [(z,'z'),(g,'g'),(j,'j'),(k,'k'),(v,'v')]:
 for b,nb in [(z,'z'),(g,'g'),(j,'j'),(k,'k'),(v,'v')]:
  if na>nb: continue
  for r in range(4): groups[na+nb+str(r)]=a*roll(b,r)
  groups[na+nb+'opp']=a[:,:2]*b[:,2:];groups[na+nb+'all']=np.column_stack([a[:,i]*b[:,l] for i,l in itertools.product(range(4),repeat=2) if na!=nb or i<=l])
for nm,a in [('z',z),('g',g),('j',j),('k',k),('v',v)]:
 for w in (.7,1.25,2.15,3.):
  for fn,f in [('sin',np.sin),('cos',np.cos),('erf',erf),('th',np.tanh),('sp',lambda a:np.logaddexp(0,a))]:groups[fn+nm+str(w)]=f(w*a)
fold=[(np.arange(52),np.arange(52,n))]+[(p[20:],p[:20]) for p in [np.random.default_rng(s).permutation(n) for s in range(1,5)]]
def cv(a,lam=0):
 scores=[]
 for tr,te in fold:
  xx=a[tr]; yy=y[tr]; scale=np.maximum(xx.std(0),.02); scale[0]=1; xx=xx/scale
  c=L(xx,yy) if not lam else np.linalg.solve(xx.T@xx+lam*np.eye(xx.shape[1]),xx.T@yy)
  err=abs(a[te]/scale@c-y[te]);scores.append([err[:,0].mean(),err[:,1:].mean()]);
 return np.array(scores)
def rank(a):
 v=cv(a);return v[:,0].mean(),v[0,0],v[:,0].max(),v[:,1].max()
print('core',cv(core)); B=[]
for key in ('','j','k','jk'):
 a=A(z,g,v,*[groups[i] for i in key]);print('base',key,cv(a).round(4).tolist());
 for name,b in groups.items():B.append((rank(np.c_[a,b]),key,name))
print('single',sorted(B)[:30]);
C=[]
for _,key,name in sorted(B)[:20]:
 a=A(z,g,v,*[groups[i] for i in key],groups[name]);
 for name2,b in groups.items():
  if name2!=name:C.append((rank(np.c_[a,b]),key,name,name2))
print('double',sorted(C)[:25]);
for _,key,name,name2 in sorted(C)[:5]:
 a=A(z,g,v,*[groups[i] for i in key],groups[name],groups[name2]); print(key,name,name2,[(l,cv(a,l).round(4).tolist()) for l in (0,.001,.01,.1)])
# Check whether an endpoint-only residual is a simple function of the cyclic product or another single variable.
for key in ('','j','k','jk'):
 a=A(z,g,v,*[groups[i] for i in key]); res=y-a@L(a,y); print('res',key,np.std(res,0).round(5),np.linalg.svd(res,compute_uv=False).round(4));
 print('scalar residual order',np.sort(res[:,0]).round(3))
