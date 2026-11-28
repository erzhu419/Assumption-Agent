exec(open('/tmp/profile01.py').read().split('ids=np.arange')[0])
from scipy.optimize import minimize_scalar
from itertools import product
p=s['panel_b']; u=p['u']; w=p['v']; print('panel B linear LOO',loo(affine(u),w)); print('panel B quad LOO',loo(base(u),w)); print('panel B residual std',np.std(w-affine(u)@fit(affine(u),w),axis=0))
a=s['panel_a']; aa=[]; bb=[]
for k,l in zip(a['u'].T,a['v'].T):
 i,j=np.triu_indices(len(k),1); d=k[i]-k[j]; keep=abs(d)>1e-9; m=np.median((l[i]-l[j])[keep]/d[keep]); aa.append(m);bb.append(np.median(l-m*k))
raw=(s['train']['x']-bb)/aa
families={'lin':(lambda x,k:affine(x),(.1,3)), 'quad':(lambda x,k:base(x),(.1,3)), 'diag2':(lambda x,k:F(x,x,x*x),(.1,3)), 'diag3':(lambda x,k:F(x,x,x**3),(.1,3)), 'tanh':(lambda x,k:F(x,x,np.tanh(k*x)),(.1,4)), 'atan':(lambda x,k:F(x,x,np.arctan(k*x)),(.1,4)), 'log':(lambda x,k:F(x,x,np.log1p(k*x*x)),(.02,4)), 'rat':(lambda x,k:F(x,x,x*x/(1+k*x*x)),(.02,4)), 'sin':(lambda x,k:F(x,x,np.sin(k*x)),(.1,4)), 'cos':(lambda x,k:F(x,x,np.cos(k*x)),(.1,4))}
ids=np.arange(len(y)); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(len(y)),6)]
for key in ['panel_b','panel_c/v1','panel_c/v2']:
 if key!='panel_b':u=c['u'];w=c[key.split('/')[1]]
 scores=[]
 for name,(fn,bounds) in families.items():
  def loss(k):
   U=fn(u,k); return np.mean((U@fit(U,w)-w)**2)
  k=minimize_scalar(loss,bounds=bounds,method='bounded').x; U=fn(u,k); scores.append((float(np.mean(loo(U,w))),name,k,loss(k)**.5))
 print(key,sorted(scores),flush=True)
 for _,name,k,_ in sorted(scores)[:3]:
  if name=='lin':continue
  fn=families[name][0]; X=fn(raw if key=='panel_b' else q,k)@fit(fn(u,k),w)
  for label,W in [('map',X),('delta_half',q+.5*(X-q))] if key=='panel_b' else [('map',X)]:
   es=[]
   for t,v in splits:
    _,g=train(W[t],y[t],'gate'); A=design(W,g,'gate'); es.append(np.mean(abs(A[v]@fit(A[t],y[t])-y[v])))
   print(key,name,label,'gate',np.round(es,5),flush=True)
