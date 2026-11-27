exec(open('/tmp/chamber_probe.py').read().split('a=cal')[0])
from itertools import combinations_with_replacement
np.set_printoptions(precision=6,suppress=True,linewidth=150)
a=cal['true_channel_loads']; b=cal['bridge_raw']; good=np.max(abs(b),axis=1)<8
print('excluded rows',np.flatnonzero(~good),'n',len(a))
for mode in ('full','sparse'):
 terms=[()]+[(j,) for j in range(4)]+list(combinations_with_replacement(range(4),2))+[(j,j,j) for j in range(4)]
 if mode=='full': terms+= [s for s in combinations_with_replacement(range(4),3) if len(set(s))>1]
 A=np.column_stack([np.prod(a[:,s],axis=1) for s in terms]); Q=np.linalg.pinv(A[good]); coef=Q@b[good]; err=A@coef-b; loo=err[good]/(1-np.sum(A[good]*Q.T,axis=1))[:,None]
 print(mode,'rank',np.linalg.matrix_rank(A[good]),'loo rmse',np.sqrt(np.mean(loo**2,axis=0)),'axial',np.sqrt(np.mean(loo.sum(1)**2))); print('excluded residuals',err[~good]);
 for term,c in zip(terms,coef):
  if np.linalg.norm(c)>0.002:print(term,c)
