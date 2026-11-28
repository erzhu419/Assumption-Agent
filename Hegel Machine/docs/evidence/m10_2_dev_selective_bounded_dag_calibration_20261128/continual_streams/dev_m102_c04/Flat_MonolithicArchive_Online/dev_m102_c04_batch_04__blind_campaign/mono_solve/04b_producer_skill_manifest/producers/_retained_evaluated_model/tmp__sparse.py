exec(open('/tmp/graph.py').read().split('rows=[]')[0])
from itertools import combinations
P,R=F1(q),F2(q); D=np.einsum('ni,nj->nij',P,R).reshape(-1,16)
G=aff(np.c_[P,R]); F=np.c_[G,D]; print('full weights: main then products'); print(ls(F[:52],Z[:52],rcond=None)[0]);
K=np.c_[D[:,::5].sum(1),D]; ids=['dot']+[(i,j) for i in range(4) for j in range(4)]
for j in range(3):
 for mode,main in [('both',G),('P',aff(P)),('R',aff(R))]:
  for size in range(4):
   def calc(idx):
    f=np.c_[main,K[:,idx]]; w=ls(f[:52],Z[:52,j],rcond=None)[0]; e=f@w-Z[:,j]; return np.sqrt(np.mean(e[:52]**2)),np.mean(abs(e[52:])),w
   best=min(combinations(range(17),size),key=lambda idx:calc(idx)[0]); out=calc(best); print(j,mode,size,[ids[i] for i in best],np.round(out[:2],6),np.round(out[2],4))
