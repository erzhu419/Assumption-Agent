import sys,pickle,numpy as np
from scipy.linalg import solve
sys.path.insert(0,'/holdout_task')
from batch_solver import _coordinates,_kernel,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; X=tr['x']; n=len(X); z=_coordinates(s,X); print('t',s['eval_times']); print('range',np.min(X,0),np.max(X,0),np.min(s['test']['x'],0),np.max(s['test']['x'],0));
splits=[(np.arange(52),np.arange(52,72))]+[(a[:52],a[52:]) for a in (np.random.default_rng(i).permutation(n) for i in range(32))]; splits+= [(np.setdiff1d(np.arange(n), a),a) for a in np.array_split(np.argsort(np.linalg.norm(X,axis=1)),4)]
def evaluate(K,ridge):
 P=[]
 for a,b in splits:
  A=solve(K[np.ix_(a,a)]+ridge*np.eye(len(a)),np.c_[y,T][a],assume_a='pos'); P.append(K[np.ix_(b,a)]@A)
 return P
def score(P):
 e=np.array([np.mean(abs(p-np.c_[y,T][b]),axis=0) for p,(a,b) in zip(P,splits)]); ey=e[:,0]; et=e[:,1:].mean(axis=1); return np.round([ey[0],ey[1:33].mean(),ey[1:33].max(),ey[33:].max(),et[0],et[1:33].mean(),et[1:33].max(),et[33:].max()],5)
rows=[]
for name,x in z.items():
 for w in [.075,.12,.18,.25,.35,.5]:
  for q in [0.,.2,1.]:
   for ridge in [.0002,.001,.003,.008,.02]:
    K=_kernel(x,w,q); P=evaluate(K,ridge); rows.append(((name,w,q,ridge),score(P),P))
print('current y',[(key,sc) for key,sc,_ in rows if key==('bi',.25,0.,.001)]);
for col in [1,5]:
 keep=[r for r in rows if r[1][0]<.038] if col==1 else rows
 key=lambda r:r[1][col]+.25*r[1][col+1]+.12*r[1][col+2]
 for a,b,_ in sorted(keep,key=key)[:12]:print(col,a,b)
 print('ens',score([np.mean(q,axis=0) for q in zip(*[r[2] for r in sorted(keep,key=key)[:20]])]))
