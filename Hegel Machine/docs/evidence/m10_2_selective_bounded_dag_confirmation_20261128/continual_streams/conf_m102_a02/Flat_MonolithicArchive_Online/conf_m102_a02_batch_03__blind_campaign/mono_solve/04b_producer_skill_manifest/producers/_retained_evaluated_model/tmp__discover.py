from probe import *
import runpy
from scipy.optimize import least_squares, minimize_scalar
m=runpy.run_path('/holdout_task/batch_solver.py'); x=m['_inverse'](s['train']['x'],m['_calibration'](s['panel_a'])); z=s['train']['traj']; y=s['train']['y']; B=m['_basis'](s['eval_times'],z[:52]); T=np.c_[y,solve(B,z.T).T];
P=s['panel_c']; U=P['u']; Y=P['v1']; n=len(U)
funs={'sin':np.sin,'tanh':np.tanh,'rat':lambda x:x/(1+x*x),'gauss':lambda x:x*np.exp(-x*x),'atan':np.arctan,'abs':abs,'sign':np.sign}
for nm,fun in funs.items():
 out=[]
 for j in range(4):
  def F(a,U=U):return np.c_[aff(U),fun(a[0]*U[:,(j+1)%4]+a[1])]
  def err(a):
   f=F(a);return f@solve(f,Y[:,j])-Y[:,j]
  rr=min((least_squares(err,[a,b],bounds=([.03,-4],[25,4]),max_nfev=150) for a in [1,3,6,10,15,20] for b in [0,1]),key=lambda r:np.linalg.norm(r.fun)); out.append((np.std(rr.fun),*rr.x,solve(F(rr.x),Y[:,j])[-1]))
 print(nm,np.array(out))
print('panel C sample',np.c_[U,Y][:12]);print('exact cal quad weights',solve(pp(x)[:52],T[:52]));
O=[np.arange(72)]+[np.random.default_rng(k).permutation(72) for k in range(6)]+[np.argsort(g) for g in [np.linalg.norm(x,axis=1),*x.T,*(-x).T]]
rows=[]
def ev(nm,F,reg=.005):
 scores=[]
 for o in O:
  i,j=o[:52],o[52:]; w=solve(np.r_[F[i],np.eye(F.shape[1])*np.sqrt(reg)],np.r_[T[i],np.zeros((F.shape[1],3))]); r=F[j]@w-T[j]; scores.append([abs(r[:,0]).mean(),abs(r[:,1:]@B.T).mean()]);
 scores=np.array(scores);rows.append((nm,np.round(scores[0],4),np.round(scores[:7].max(0),4),np.round(scores[7:].max(0),4),np.round(scores[7:].mean(0),4)))
for nm,fun in funs.items():
 for a in [.5,1,1.5,2,3,5,8]:
  g=fun(a*x); hx=g*x; cross=(x[:,:,None]*g[:,None,:]).reshape(72,-1); G=[np.c_[pp(x),g],np.c_[pp(x),g,hx],np.c_[pp(x),g,cross],np.c_[aff(x),g,hx]]
  for j,F in enumerate(G):ev((nm,a,j),F)
print('main best');print(*sorted(rows,key=lambda r:sum(r[1]))[:12],sep='\n');print('tail best');print(*sorted(rows,key=lambda r:sum(r[3]))[:16],sep='\n')
