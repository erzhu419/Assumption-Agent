exec(open('/tmp/probe.py').read().split('results={}')[0])
from itertools import combinations
v=s['panel_c']['v2']; us=[('u',u)]
for r in (1,-1,2):
 w=np.roll(u,r,axis=1)
 us.extend([(f'u+{r}',u+w),(f'u-{r}',u-w),(f'u*{r}',u*w)])
for k in ('v1','v2'):
 v=s['panel_c'][k]; R=[]
 def err(F):
  h=_bias(F); return (h@_ls(h,v)-v).ravel()
 def fit(m,p,b):
  q=least_squares(lambda p:err(m(p)),p,bounds=b,max_nfev=50).x
  return np.std(err(m(q))),q
 for nm,t in us:
  for a,fa in ff.items():
   m=lambda p:fa(p[0]*t)
   e,p=fit(m,[1.5],([.02],[5.])); R.append((e,nm,a,p))
   if nm!='u':
    for b in ('linear','sin','cos','tanh','sat','quad','rational'):
     fb=ff[b]; m=lambda p:np.column_stack([fa(p[0]*u),fb(p[1]*t)])
     e,p=fit(m,[1.4,1.2],([.02,.02],[5.,5.]));R.append((e,nm,a+'|'+b,p))
   else:
    for b,fb in ff.items():
     m=lambda p:np.column_stack([fa(p[0]*u),fb(p[1]*u)])
     e,p=fit(m,[1.4,1.2],([.02,.02],[5.,5.]));R.append((e,nm,a+'|'+b,p))
 print(k,*sorted(R,key=lambda r:r[0])[:15],sep='\n',flush=True)
