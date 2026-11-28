exec(open('/tmp/explore.py').read().split('for fn in')[0])
from scipy.optimize import minimize_scalar
from itertools import combinations
zt=n.linalg.solve(b[1:].T,(s['test']['x']-b[0]).T).T
FNS={'tanh':n.tanh,'sin':n.sin,'rat':lambda h:h/(1+h*h),'atan':n.arctan,'asinh':n.arcsinh,'soft':lambda h:h/n.sqrt(1+h*h),'erf':__import__('scipy.special',fromlist=['erf']).erf}
def opfit(u,v,fn,linear):
 def des(q,w):
  h=fn(w*n.exp(q));return n.c_[n.ones(len(w)),w,h] if linear else n.c_[n.ones(len(w)),h]
 def obj(q):
  D=des(q,u);return (D@ls(D,v)-v).ravel()
 opts=[least_squares(obj,n.full(4,n.log(a)),bounds=(-2,2),max_nfev=100) for a in [.65,1.4,2.5,3.8]]
 o=min(opts,key=lambda o:n.linalg.norm(o.fun)); D=des(o.x,u); c=ls(D,v); hat=n.sum(D*n.linalg.pinv(D).T,1); err=n.sqrt(n.mean((o.fun.reshape(v.shape)/(1-hat[:,None]))**2));return err,lambda w:des(o.x,w)@c,n.exp(o.x),c
ops={}
for key,k in [('panel_b','v'),('panel_c','v1')]:
 p=s[key]; arr=[]
 for name,fn in FNS.items():
  for lin in [False,True]:
   e,op,q,c=opfit(p['u'],p[k],fn,lin);arr.append((e,name,lin,op)); print(k,name,lin,round(e,5),q,'lin norm',n.linalg.norm(c[1:5]) if lin else None)
 ops[k]=sorted(arr,key=lambda a:a[0])[:3]
p=s['panel_c']; ds=lambda x:n.c_[n.ones(len(x)),x,x*x]; c2=ls(ds(p['u']),p['v2']); V2=lambda x:ds(x)@c2
print('v2 sq rmse',n.std(ds(p['u'])@c2-p['v2']))
def score(D,idx=None):
 if idx is None:idx=n.arange(52)
 te=n.setdiff1d(n.arange(len(z)),idx); C=ls(D[idx],Y[idx]); E=n.mean(abs(D[te]@C-Y[te]),0);return E[[0,1,8]]
rows=[]
for i,(_,bn,bl,B) in enumerate(ops['v']):
 for j,(_,cn,cl,C) in enumerate(ops['v1']):
  W={'z':z,'Bz':B(z),'Cz':C(z),'Vz':V2(z),'BCz':B(C(z)),'CBz':C(B(z)),'VBz':V2(B(z)),'BVz':B(V2(z)),'VCz':V2(C(z)),'CVz':C(V2(z))}
  for key,w in W.items():
   for d in [1,2,3]:
    D=poly(w,d); rows.append((score(D),i,j,key,'p'+str(d)))
   for fn in [n.tanh,n.arcsinh,n.sin]:
    for a in [.5,1,2,3.5,5]:
     h=fn(a*w)
     for key2,D in [('add',n.c_[poly(w,1),h,h*h,w*h]),('quad',n.c_[poly(h,2),w,w*h]),('mix',n.c_[poly(w,2),h,h*h,w*h])]:
      rows.append((score(D),i,j,key,(fn.__name__,a,key2)))
  for keys in combinations(W,2):
   D=n.c_[n.ones(len(z)),W[keys[0]],W[keys[1]],W[keys[0]]**2,W[keys[1]]**2];rows.append((score(D),i,j,keys,'pair'))
for col in range(3):
 print('best target',col);print(*sorted(rows,key=lambda a:a[0][col])[:10],sep='\n')
