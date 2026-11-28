import pickle,numpy as np,itertools
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; n=52; A=np.load('/tmp/cached.npz'); z=A['z']; b=A['panel_bvlinear']; c=A['panel_cv1soft']; d=A['panel_cv2dead']; t=s['train']['traj'];
def fit(X,ys):return np.linalg.lstsq(X,ys,rcond=None)[0]
def check(h):
 H=h[:n]; v=H@fit(H,y[:n])-y[:n]; q=np.linalg.svd(H,full_matrices=False)[0]; dof=(q*q).sum(axis=1); return np.mean(abs(v/np.maximum(1-dof,1e-5))),np.mean(abs(h[n:]@fit(H,y[:n])-y[n:])),np.sqrt(np.mean(v*v))
P={'z':z,'b':b,'c':c,'d':d}; F={}; q,r=np.triu_indices(4)
for a,w in P.items():
 F[a]=w; F[a+'q']=w[:,q]*w[:,r]; F[a+'pair']=(w[:,0]*w[:,2]+w[:,1]*w[:,3])[:,None]; F[a+'cube']=w**3
 for i in range(4):
  for j in range(4):
   for k in range(j,4):
    if i>j:continue
    if k==j or j==i:continue
for a,w in P.items():
 for k,v in P.items():
  if a==k:continue
  F[a+'/'+k]=(w/(1+.22*v*v)); F[a+'roll/'+k]=(w/(1+.22*np.roll(v,2,axis=1)**2)); F[a+'x'+k]=(w[:,:,None]*v[:,None,:]).reshape(-1,16); F[a+'dot'+k]=(w*v).sum(1)[:,None]
res=[]
for base in [('b',),('c',),('b','c'),('b','c','d')]:
 beam=[base]; seen=set(beam)
 for depth in range(3):
  cur=[]
  for seq in beam:
   for k in F:
    new=tuple(sorted(set(seq+(k,))))
    if new in seen:continue
    seen.add(new); h=np.column_stack([np.ones(len(y))]+[F[i] for i in new]);
    if h.shape[1]>39 or np.linalg.matrix_rank(h[:n])<h.shape[1]:continue
    stats=check(h); cur.append((stats,new))
  cur.sort(); res+=cur[:8]; beam=[n for _,n in cur[:10]]
print('best LOO:');print(*sorted(set(res))[:24],sep='\n'); print('trajectory x,z,B,C,D',*[np.abs(np.c_[np.ones(len(y)),w][n:]@fit(np.c_[np.ones(len(y)),w][:n],t[:n])-t[n:]).mean() for w in P.values()])
