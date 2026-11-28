exec(open('/tmp/panels.py').read().split("for p in ['panel_b'")[0])
from scipy.optimize import minimize_scalar
from itertools import combinations

def basis(x, terms, w=1.):
 b=[np.ones((len(x),1)),x]
 for t in terms:
  if t=='q': b.append(x*x)
  elif t=='p': b.append(np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2)]))
  elif t=='s': b.append(np.sin(w*x))
  elif t=='c': b.append(np.cos(w*x))
  elif t=='h': b.append(np.tanh(w*x))
  elif t=='z': b.append(x**3)
 return np.column_stack(b)
def loo(A,Y):
 Q,R=np.linalg.qr(A); E=(Y-Q@(Q.T@Y))/np.maximum(1-(Q*Q).sum(1),1e-9)[:,None]; return np.abs(E).mean(0)
models={}
for p,v in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 x,y=s[p]['u'],s[p][v]; fits=[]
 for t in ['', 'q','qp','z','qz','qpz','s','qs','qps','c','qc','qpc','h','qh','qph']:
  def f(w): return loo(basis(x,t,w),y).mean()
  w=1.
  if any(k in t for k in 'sch'):
   g=np.linspace(.6,6,55); k=np.argmin([f(w) for w in g]); w=minimize_scalar(f,bounds=(g[max(0,k-1)],g[min(len(g)-1,k+1)]),method='bounded').x
  fits.append((f(w),t,w))
 print(p,v,[(round(e,5),t,round(w,4)) for e,t,w in sorted(fits)])
 models[v]=sorted(fits)[:2]
a=s['panel_a'];good=np.max(abs(a['v']-a['u']),axis=1)<1;u=a['u'][good];v=a['v'][good];g=((u-u.mean(0))*(v-v.mean(0))).sum(0)/((u-u.mean(0))**2).sum(0);o=v.mean(0)-g*u.mean(0);x=(s['train']['x']-o)/g;Y=np.column_stack([s['train']['y'],s['train']['traj']]); maps={}
for p,v in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 for k,(_,t,w) in enumerate(models[v]):
  A=basis(s[p]['u'],t,w); C=np.linalg.lstsq(A,s[p][v],rcond=None)[0];maps[v+str(k)]=basis(x,t,w)@C
for ks in [(),('v0',),('v10',),('v20',),('v10','v20'),('v0','v10'),('v0','v20'),('v0','v10','v20'),('v1','v11','v21')]:
 M=[maps[k] for k in ks]; X=np.column_stack([np.ones(len(x)),*M]) if M else basis(x,'');
 for add in ['', 'x','s','q','h']:
  A=np.column_stack([X,{'x':x,'s':np.sin(2*x),'q':x*x,'h':np.tanh(2*x),'':np.empty((len(x),0))}[add]])
  e=loo(A,Y); print(ks,repr(add),np.round([e[0],e[1:].mean(),e[-1]],5))
