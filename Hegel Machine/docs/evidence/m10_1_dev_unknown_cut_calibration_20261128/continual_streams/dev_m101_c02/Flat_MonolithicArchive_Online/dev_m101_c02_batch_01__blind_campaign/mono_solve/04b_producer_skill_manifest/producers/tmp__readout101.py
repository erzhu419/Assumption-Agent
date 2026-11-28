exec(open('/tmp/graph101.py').read().split('for key,p,v in')[0])
import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _fit
from scipy.special import erf
from itertools import combinations_with_replacement

def cubic(z):return np.column_stack([np.prod(z[:,p],axis=1) for p in combinations_with_replacement(range(4),3)])
def D(z):return np.column_stack([np.prod(z[:,p],axis=1) for p in combinations_with_replacement(range(4),2)])
H=lambda z:np.c_[z,np.tanh(z),np.tanh(2*z),np.tanh(4*z),z**3]
# Stress inputs in the tails as well as repeat random 56/16 splits.
norm=np.linalg.norm((x-off)/sc,axis=1); order=np.argsort(norm); folds += [(order[:-12],order[-12:]),(order[12:],order[:12])]
results={}; coef=[]
for i,j in folds:
 predict, coord=_fit(s,x[i],Y[i,0],Y[i,1:]); z=coord(x); base=np.column_stack(predict(x)); E=Y[:,1:]-base[:,1:]; _,dv,T=np.linalg.svd(E[i],full_matrices=False); Q=O(z); target=E@T[0]; coef.append((dv[:2],T[0],L(Q[i],Y[i,1:]@T[0]))); values={'base':base}
 for kind,B in [('z2',D(z)),('z3',cubic(z)),('z23',np.c_[D(z),cubic(z)]),('w2',D(w if 'w' in globals() else (x-off)/sc)),('znon',H(z)),('w', (x-off)/sc),('pair',z[:,[0,1]]*z[:,[2,3]])]:
  B=B-Q@L(Q[i],B[i]); B=B/np.maximum(np.std(B[i],axis=0),1e-9)
  for alpha in [0.1,3,20]:
   c=np.linalg.solve(B[i].T@B[i]+alpha*np.eye(B.shape[1]),B[i].T@target[i]); values[kind+str(alpha)]=np.c_[base[:,0],base[:,1:]+np.outer(B@c,T[0])]
 for k,p in values.items():
  er=abs(p[j]-Y[j]); results.setdefault(k,[]).append([er[:,0].mean(),er[:,1:].mean(),er[:,-1].mean(),max(er[:,1:].mean(0))]);
print('baseline and best corrections:');
for k in ['base']+sorted(results,key=lambda k: np.mean(results[k],axis=0)[1])[:14]:
 e=np.array(results[k]); print(k,'56/16',e[0],'ordinary max',e[:6].max(0),'tails',e[6:]);
print('residual sv/time/association',coef[0]);
_, coord=_fit(s,x,Y[:,0],Y[:,1:]); z=coord(x); zt=coord(s['test']['x']); print('z range',z.min(0), z.max(0),zt.min(0),zt.max(0))
