exec(open('/tmp/readout_diag.py').read().split('for keys in')[0])
from itertools import combinations_with_replacement
from scipy.linalg import solve
u=s['panel_b']['u'];v=s['panel_b']['v']; f=2.83
co=m._fit(m._b_basis(u,f),v); print('B sine to linear ratio',np.round(np.sum(co[1:5]*co[5:],axis=1)/np.sum(co[1:5]**2,axis=1),4)); print('A aligned range',z.min(0),z.max(0)); print('A map',m._align(s['panel_a'],np.r_[np.zeros((1,4)),np.eye(4)]))
B=b-co[0]; h=B@np.linalg.inv(co[1:5]); print('B unmixed and rotated range',np.max(abs(h),0),np.max(abs(B),0))
Y=np.c_[y,tr]; folds += [(np.argsort(z[:,j])[:-10],np.argsort(z[:,j])[-10:]) for j in range(4)]
def gram(A):
 A=A-A.mean(0); return A@A.T/(A.shape[1]*np.mean(A*A))
def features(A):
 A=A-A.mean(0); return A/np.sqrt(np.mean(A*A))
def radial(A,a):
 A=features(A); d=np.mean((A[:,None]-A[None,:])**2,axis=2); return np.exp(-a*d)
gj={k:gram(a) for k,a in [('z',z),('b',b),('h',h),('c',c1),('d',c2)]}
G={'z':gj['z'],'b':gj['b'],'h':gj['h'],'zb':(gj['z']+gj['b'])/2,'zh':(gj['z']+gj['h'])/2,'zbc':(gj['z']+gj['b']+.15*gj['c'])/2.15,'zbcd':(gj['z']+gj['b']+.15*(gj['c']+gj['d']))/2.3,'zbh':(.5*gj['z']+gj['b']+.5*gj['h'])/2}
blocks={'b':b,'h':h,'z':z,'c':c1,'d':c2}
def pred(K,p):
 scores=[]
 for it,iv in folds:
  V=np.linalg.svd(tr[it],full_matrices=False)[2][:2]; yt=np.c_[y[it],tr[it]@V.T]; P=K[np.ix_(iv,it)]@solve(K[np.ix_(it,it)]+p*np.eye(len(it)),yt,assume_a='pos'); scores.append([np.mean(abs(P[:,0]-y[iv])),np.mean(abs(P[:,1:]@V-tr[iv]))])
 a=np.array(scores); return a[0],a[:5].max(0),a[5:].max(0),a[:5].mean(0)
rows=[]
def add(name,K):
 for p in [1e-6,.00003,.0003,.002]:
  off,w,tail,avg=pred(K,p);rows.append((max(w)+.18*max(tail),name,p,off,w,tail,avg))
for k,L in G.items():
 for a in [.05,.15,.4]:
  for d in [2,3,4,5]:add(('poly',k,a,d),(1+a*L)**d)
for a in [.05,.1,.2,.4]:
 for w in [.1,.35,.65]:
  for k in ['b','h','c','d']:
   A=radial(blocks[k],a);Z=radial(z,a); add(('rbf',k,a,w), 1 + Z**w*A**(1-w)+.03*gj['z'])
for f in [.8,1.5,2.5,3.5,5.]:
 for k in ['b','h']:
  B=features(blocks[k]); A=gram(np.c_[np.sin(f*B),np.cos(f*B)]); L=G['zb'];
  for w in [.05,.25,1.]: add(('harm',k,f,w),(1+.2*L)**3+w*A*(1+.1*L))
for row in sorted(rows)[:16]+sorted(rows,key=lambda r:max(r[3])+.08*max(r[4]))[:8]: print(row[1:3],*[np.round(t,5) for t in row[3:]])
