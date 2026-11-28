from probe_graph import *
from sklearn.linear_model import Lasso, MultiTaskLasso
np.set_printoptions(precision=5,suppress=True,linewidth=140)
z=ops(S,T['x']); c,d=z['c'],z['d']; X=np.c_[c,d]; F=poly(X,2); names=['1']+[a+str(i) for a in 'cd' for i in range(4)]; names += [a+'*'+b for a,b in itertools.combinations_with_replacement(names[1:],2)]
splits=[(np.arange(52),np.arange(52,72))]+[(p[:56],p[56:]) for p in [np.random.default_rng(k).permutation(72) for k in range(6)]]
blocks=[list(range(1,5)),list(range(5,9)),[i for i,n in enumerate(names) if '*c' in n and n.startswith('c')],[i for i,n in enumerate(names) if '*d' in n and n.startswith('c')],[i for i,n in enumerate(names) if '*d' in n and n.startswith('d')]]
def out(p,v):return np.array([np.mean(abs(p[:,0]-Y[v,0])),np.mean(abs(p[:,1:]-Y[v,1:]))])
for select in itertools.product([0,1],repeat=3):
 ids=[0]+sum(blocks[:2]+[b for use,b in zip(select,blocks[2:]) if use],[]);G=F[:,ids]
 scores=[]
 for a,b in splits:
  sc=np.maximum(.1,G[a].std(0));sc[0]=1;A=G/sc;scores.append(out(ridge(A[b],A[a],Y[a],.0003),b))
 print('blocks cc/cd/dd',select,'fixed',scores[0],'worst',np.max(scores,0),'mean',np.mean(scores,0))
a,b=splits[0]; V=np.linalg.svd(T['traj'][a],full_matrices=False)[2][:2];Q=np.c_[T['y'],T['traj']@V.T]; sc=F[a].std(0);sc[0]=1; A=F/sc
for j in range(3):
 print('target',j)
 for alpha in [.01,.003,.001,.0003,.0001]:
  m=Lasso(alpha=alpha,max_iter=30000,tol=1e-7).fit(A[a,1:],Q[a,j]); ix=np.r_[0,np.where(abs(m.coef_)>1e-8)[0]+1];p=A[:,ix]@ls(A[a][:,ix],Q[a,j]);print(alpha,len(ix),np.mean(abs(p[b]-Q[b,j])),np.sqrt(np.mean((p[a]-Q[a,j])**2)),[(names[i],round(m.coef_[i-1],4)) for i in ix[1:] if abs(m.coef_[i-1])>.015])
coef=ls(F[a],Q[a]); print('full quad training RMS',np.sqrt(np.mean((F[a]@coef-Q[a])**2,axis=0)));print('shared temporal vectors',V)
