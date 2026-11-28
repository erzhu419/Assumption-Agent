import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
A=np.load('/tmp/paths.npz'); z=A['z']; x=A['x']; T=A['T']; Y=A['Y']; t=A['t']; C=A['c1']; n=52
fns={'exp':lambda k:-np.expm1(-k*t),'rat':lambda k:t/(1+k*t),'log':lambda k:np.log1p(k*t),'tanh':lambda k:np.tanh(k*t),'sqrt':lambda k:np.sqrt(1+k*t)-1,'atan':lambda k:np.arctan(k*t)}
from itertools import combinations_with_replacement as cwr
results=[]
for a,b in cwr(fns,2):
 def basis(k):return np.column_stack((fns[a](k[0]),fns[b](k[1])))
 def err(k):
  Q=basis(k); return (Q@np.linalg.lstsq(Q,T[:n].T,rcond=None)[0]-T[:n].T).ravel()
 opts=[least_squares(err,p,bounds=(.01,12.)) for p in ([.25,1.25],[1.25,.22],[1.25,1.7],[1.8,.4])]
 q=min(opts,key=lambda q:np.mean(q.fun**2)); Q=basis(q.x); p=(Q@np.linalg.lstsq(Q,T[n:].T,rcond=None)[0]).T
 results.append((np.sqrt(np.mean(q.fun**2)),np.mean(abs(T[n:]-p)),a,b,q.x))
for r in sorted(results)[:15]:print('time',r)
cs=[c for k in range(1,4) for c in combinations_with_replacement(range(4),k)]
for name,u in [('z',z),('x',x)]:
 D=np.column_stack([np.ones(len(u))]+[np.prod(u[:,c],axis=1) for c in cs]); w=np.linalg.lstsq(D[:n],np.c_[Y,A['S']][:n],rcond=None)[0]
 print('terms',name); print('\n'.join(str((c,np.round(q,5))) for c,q in zip([()]+cs,w) if max(abs(q))>.012))
# Compare supported feature blocks over fixed and rotated holdouts. Remove exact linear dependencies by SVD.
P=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
mono=lambda a,k:np.column_stack([np.prod(a[:,c],axis=1) for c in cwr(range(a.shape[1]),k)])
O=np.ones((len(x),1)); ij=[(0,1),(0,3),(1,2),(2,3)]; bil=np.column_stack([z[:,i]*z[:,j] for i,j in ij]); vc=A['c2'];
base=np.c_[O,z,bil]; ds={'base':base,'base + zbil':np.c_[base,P(z,bil)],'base + xbil':np.c_[base,P(x,bil)],'base + bil^2':np.c_[base,mono(bil,2)],'base + zbil + bil^2':np.c_[base,P(z,bil),mono(bil,2)],'base + v2':np.c_[base,vc], 'base + v2 + zbil':np.c_[base,vc,P(z,bil)],'base + v2 + zv2':np.c_[base,vc,P(z,vc)],'base + v2 + v2bil':np.c_[base,vc,P(vc,bil)],'z cubic':np.c_[O,z,mono(z,2),mono(z,3)],'z quadratic':np.c_[O,z,mono(z,2)]}
rng=np.random.default_rng(159); splits=[np.arange(72)]+[rng.permutation(72) for i in range(16)]; target=np.c_[Y,T];
for name,D in ds.items():
 scores=[]
 for split in splits:
  tr=split[:52];va=split[52:]; norm=D[tr].std(0);norm[norm<1e-8]=1; d=D/norm; u,s,v=np.linalg.svd(d[tr],full_matrices=False); w=(v.T*(s/(s*s+.002)))@u.T@target[tr]; e=abs(target[va]-d[va]@w);scores.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()]);
 w=np.linalg.lstsq(D[:52],target[:52],rcond=1e-8)[0]; print(name,'rank',np.linalg.matrix_rank(D),'fit',abs(D[:52]@w-target[:52]).mean(0)[[0,-1]],'fixed',np.round(scores[0],5),'mean',np.round(np.mean(scores,axis=0),5),'max',np.round(np.max(scores,axis=0),5))
