exec(open('/tmp/sparse_probe.py').read().split('for nm,fn in variants')[0])
q=-np.expm1(-t[:,None]*[.4125,1.80995]); A=np.c_[Y[:,0],fit(q,Y[:,1:].T).T];
for name,w in [('z',z),('b',aff(z)@R),('bi',(z-R[0])@np.linalg.inv(R[1:]))]:
 for nm,f in [('cos24',np.cos(2.4*w)),('atan22',np.arctan(2.2*w)),('both',np.c_[np.cos(2.4*w),np.arctan(2.2*w)]),('lin_cos',np.c_[w,np.cos(2.4*w)]),('combined',np.c_[w,np.cos(2.4*w),np.arctan(2.2*w)])]:
  h=aff(f); c=fit(h[:52],A[:52]); r=A-h@c; print(name,nm,np.sqrt(np.mean(r[:52]**2,axis=0)).round(5),np.mean(abs(r[52:]),axis=0).round(5));
  if name=='z': print(c.round(3).T)
print('amplitude std',A.std(0))
# Broad additive structure: systematically compare coordinates and exact family blocks.
from itertools import product
z1=z; z2=aff(z)@R; z3=(z-R[0])@np.linalg.inv(R[1:]);
for k in (1,2,3,4):
 for j in range(3):
  groups={}
  for i,w in enumerate((z1,z2,z3)):
   groups.update({f'{i}L':w, f'{i}Q':w*w,f'{i}C':np.cos(2.4*w),f'{i}S':np.sin(2.4*w),f'{i}T':np.arctan(2.2*w),f'{i}X':np.column_stack([w[:,a]*w[:,b] for a,b in combinations(range(4),2)]),f'{i}P':np.column_stack([np.prod(w[:,idx],axis=1) for idx in combinations(range(4),3)])})
  results=[]
  for ix in combinations(groups,k):
   h=aff(np.column_stack([groups[i] for i in ix])); c=fit(h[:52],A[:52,j]); r=A[:,j]-h@c; results.append((np.linalg.norm(r[:52])/np.sqrt(52),np.mean(abs(r[52:])),ix))
  print('blocks',j,k, 'train',sorted(results)[:2],'cv',min(results,key=lambda x:x[1]))
