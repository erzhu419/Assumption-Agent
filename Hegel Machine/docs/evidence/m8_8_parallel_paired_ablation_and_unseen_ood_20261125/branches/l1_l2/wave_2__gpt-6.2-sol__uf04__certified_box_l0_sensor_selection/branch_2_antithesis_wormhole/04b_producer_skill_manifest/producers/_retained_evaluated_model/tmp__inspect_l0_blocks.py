import glob,pickle,numpy as np
from scipy.linalg import eigvalsh
for f in sorted(glob.glob('/tmp/l0_failed_*.pkl')):
 a=pickle.load(open(f,'rb')); X=a['X']; H=X.T@X+a['ridge']*np.eye(X.shape[1]); s=np.sqrt(H.diagonal()); C=H/s[:,None]/s[None,:]; V=np.abs(C.copy()); np.fill_diagonal(V,0); groups=[]; unused=set(range(len(s)))
 while unused:
  j,k=np.unravel_index(V.argmax(),V.shape)
  if V[j,k]<.85: groups.extend([[i] for i in sorted(unused)]); break
  groups.append([j,k]); unused.difference_update((j,k)); V[[j,k],:]=0;V[:,[j,k]]=0
 B=np.zeros_like(C)
 for g in groups: B[np.ix_(g,g)]=C[np.ix_(g,g)]
 print(f.split('_failed_')[1], 'pairs',sum(len(g)==2 for g in groups),'diag floor',round(eigvalsh(C)[0],6),'block fraction',round(eigvalsh(C,B)[0],6))
