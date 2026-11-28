exec(open('/tmp/compose.py').read().split('for name,c in base.items():')[0])
from scipy.linalg import qr
from itertools import product
_,_,vt=np.linalg.svd(tr[:52],full_matrices=False); target=tr@vt[:2].T
maps={'B':B,'C':C,'D':D}; vals={'u':z}
for depth in range(3):
 for n,v in list(vals.items()):
  if len(n)!=depth+1:continue
  for k,m in maps.items():vals[n+k]=m(v)
results=[]; feats={}
def add(n,v):
 if not np.isfinite(v).all(): return
 a=aff(v); p=np.linalg.lstsq(a[:52],target[:52],rcond=None)[0]; e=np.mean(abs((a[52:72]@p-target[52:])@vt[:2])); t=np.mean(abs((a[:52]@p-target[:52])@vt[:2])); results.append((e,t,n));feats[n]=v
for n,v in vals.items():add(n,v)
small={n:v for n,v in vals.items() if len(n)<=3};
for n,v in small.items():
 for k,w in small.items():
  for op,q in [('mul',v*w),('rat',v/(1+w*w)),('gate',v*(1+np.tanh(w))),('add',v+w),('sub',v-w),('sqadd',v+w*w),('cubeadd',v+w**3)]:
   add((n,k,op),q)
   if len(n)+len(k)<=4:
    for mn,m in maps.items():add((n,k,op,mn),m(q))
print('BEST',sorted(results,key=lambda r:r[:2])[:30]); print('TRAIN',sorted(results,key=lambda r:r[1])[:20]);
for e,t,n in sorted(results,key=lambda r:r[:2])[:10]:print(n,score(feats[n]))
# Identify whether a compact flexible model on calibrated channels suffices.
from itertools import combinations_with_replacement
for degree in range(1,7):
 a=np.column_stack([np.prod(z[:,ix],axis=1) for d in range(0,degree+1) for ix in combinations_with_replacement(range(4),d)]); 
 for reg in [1e-5,.001,.01,.1,1]:
  p=a[:52].T@np.linalg.solve(a[:52]@a[:52].T+reg*np.eye(52),np.c_[y[:52],target[:52]]); r=a[52:72]@p-np.c_[y[52:],target[52:]];print('poly',degree,reg,np.mean(abs(r[:,0])),np.mean(abs(r[:,1:]@vt[:2])))
