import numpy as np,pickle,runpy
from itertools import combinations_with_replacement
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];x=t['x']; y=t['y'];z=t['traj'];Y=np.column_stack((y,z));frozen=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})
np.set_printoptions(precision=4,suppress=True,linewidth=160)
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for n in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),n)])
def squash(x,a):return x/np.sqrt(1+a*x*x)
a=s['panel_a'];u,v=a['u'],a['v'];A=aff(u); ii=np.argsort(np.linalg.norm(v-u,axis=1))[:-3];M=np.linalg.lstsq(A[ii],v[ii],rcond=None)[0]; print('A robust',M,'res',np.sort(np.linalg.norm(A@M-v,axis=1)));print('a few\n',np.c_[u,v][:22]); print('B\n',np.c_[s['panel_b']['u'],s['panel_b']['v']][:12]);c=s['panel_c']; C=[np.linalg.lstsq(aff(c['u']),c['v'+str(j)],rcond=None)[0] for j in [1,2]]; print('C singular',*[np.linalg.svd(m[1:],compute_uv=False) for m in C]); print('SVD temporal',np.linalg.svd(z,full_matrices=False)[2][:2]);
folds=[(i,j) for r in range(3) for i,j in KFold(6,shuffle=True,random_state=r+12).split(x)]
def score(F,p):
 E=[]
 for i,j in folds:
  G=F[i];b=np.linalg.solve(G.T@G+p*np.eye(F.shape[1]),G.T@Y[i]);E.append(np.abs(F[j]@b-Y[j]).mean(0))
 E=np.array(E);return np.array([E[:,0].mean(),E[:,1:].mean(),E[:,0].max(),E[:,1:].mean(1).max()])
xc=np.linalg.solve(M[1:].T,(x-M[0]).T).T
Xs={'x':x,'full-cal':xc,'diag-cal':(x-M[0])/M[1:].diagonal()}; features={k:q[:len(x)] for k,q in frozen.items()}
for k,q in Xs.items():
 for name,w in [('identity',q)]+[(f'C{j+1}',aff(q)@m) for j,m in enumerate(C)]:
  for d in (1,2,3,4):
   for a in (0,.04,.09,.2,.5):features[k+name+str(d)+'_'+str(a)]=poly(squash(w,a),d)
for k in ('x','full-cal','diag-cal'):
 features[k+'multiplex']=np.column_stack([poly(squash(aff(Xs[k])@m,.09),3) for m in C]+[poly(Xs[k],3)])
r=[]
for k,F in features.items():
 for p in (.00001,.0003,.003,.015,.06,.2):r.append((score(F,p),k,p))
for o in (0,1):
 print('top',o)
 for q,k,p in sorted(r,key=lambda h:h[0][o])[:18]:print(k,p,q)
print('frozen/archive-like');print([(k,score(F,.01)) for k,F in features.items() if k in frozen]);
