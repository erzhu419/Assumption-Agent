exec(open('/tmp/probe.py').read().split("a=s['panel_a']")[0])
from scipy.optimize import least_squares
from itertools import combinations
p=s['panel_a']; g=np.linalg.norm(p['v'],axis=1)<5; u=p['u'][g];v=p['v'][g]; terms=[()]+[j for d in (1,2,3) for j in C(range(4),d)]; m=fit(poly(u,3),v); print('A exact?',np.max(abs(poly(u,3)@m-v))); print([(j,c) for j,c in zip(terms,m) if max(abs(c))>1e-6]);
for k in ['panel_b','panel_c']:
 p=s[k];u=p['u']; X=poly(u,1)
 for j,y in p.items():
  if j=='u': continue
  print(k,j)
  for power in [2,3]:
   U=u**power
   def calc(q):
    Z=poly(u+q*U,1); return Z@fit(Z,y)-y
   sol=least_squares(lambda q:calc(q).ravel(),np.full(4,.4 if power==2 else -.2));print('mixed power',power,sol.x,np.std(sol.fun))
  for f in [np.tanh,np.sin,np.sinh,lambda s:s/np.sqrt(1+s*s),lambda s:s/(1+s*s),np.expm1]:
   def calc(q):
    Z=poly(f(u*np.exp(q[:4])+q[4:]),1); return Z@fit(Z,y)-y
   sol=least_squares(lambda q:calc(q).ravel(),np.r_[np.full(4,-.2),np.zeros(4)],bounds=(-2,2));print('mixed',f.__name__,sol.x,np.std(sol.fun))
  for pow in [2,3,4,5,6,7]:
   Z=np.column_stack([X]+[u**d for d in range(2,pow+1)]);print('separable',pow,np.std(Z@fit(Z,y)-y))
  print('sparse polynomial forward selection')
  Z=poly(u,3); keep=list(range(5)); rem=list(range(5,35));
  for _ in range(10):
   err,j=min((np.linalg.norm(Z[:,keep+[i]]@fit(Z[:,keep+[i]],y)-y),i) for i in rem);keep.append(j);rem.remove(j);print(terms[j],np.round(err,8),end='; ')
  print('\n',[(terms[i],r) for i,r in zip(keep,fit(Z[:,keep],y))]);
