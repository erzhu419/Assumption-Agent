exec(open('/tmp/grammar.py').read().split('rows=[]')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
np.set_printoptions(precision=4,suppress=True)
N=52
for key in ('x','a','a1'):
 z=trans[key][:72]; P=np.column_stack([np.prod(z[:,j],axis=1) for k in range(4) for j in cwr(range(4),k)]); W=fit(P[:N],Y[:N]); print(key, list(zip([j for k in range(4) for j in cwr(range(4),k)],W.round(3))))
S=s['train']; Q=Q if 'Q' in globals() else -np.expm1(-s['eval_times'][:,None]*[.392,1.857]);
folds=[np.arange(N,72),*np.array_split(np.arange(72),4),*[np.argsort(S['x'][:,j])[-15:] for j in range(4)]]
rows=[]
def poly(z,d): return np.column_stack([np.prod(z[:,j],axis=1) for k in range(d+1) for j in cwr(range(4),k)])
def score(F,reg,it,iv):
 f=F[it]; mu=f.mean(0); sd=np.maximum(f.std(0),1e-7); f=(f-mu)/sd; g=(F[iv]-mu)/sd; w=np.linalg.solve(f@f.T+reg*np.eye(len(f)),Y[it]-Y[it].mean(0)); return g@f.T@w+Y[it].mean(0)
def metrics(y,iv):return [np.mean(abs(y[:,0]-Y[iv,0])),np.mean(abs(y[:,1:]@Q.T-S['traj'][iv]))]
trans0={k:trans[k][:72] for k in ['a','ba','Ba','a1','a0']};
for k,z in list(trans0.items()):
 if k not in ['a','ba','Ba']:continue
 v=cs[1](z); w=fit(bias(z),v); u=z+(v-bias(z)@w)@np.linalg.inv(w[1:]); delta=u-z
 for g in [-2,-1,-.5,.22,.5,1,1.5,2,3]:trans0[k+'blend'+str(g)]=z+g*delta
 for g in [-1,-.5,.22,.5,1]:trans0[k+'exp'+str(g)]=z+g*np.expm1(-.296*z*z)
for k,z in trans0.items():
 for d in [2,3,4]:
  for k2 in ['','mix','tri','sat']:
   F=poly(z,d)[:,1:];
   if k2=='mix':F=np.c_[F,poly(trans0['a' if k!='a' else 'a1'],d)[:,1:]]
   if k2=='tri':F=np.c_[F,np.sin(z),np.cos(z),np.sin(2*z),np.cos(2*z)]
   if k2=='sat':F=np.c_[F,poly(np.tanh(z),d)[:,1:]]
   for reg in [.001,.015,.08,.3,1]:
    err=np.array([metrics(score(F,reg,np.setdiff1d(np.arange(72),iv),iv),iv) for iv in folds]); rows.append((k,d,k2,reg,err))
for j in range(2):
 print('target',j)
 for rank in [lambda e:e[1:5,j].mean()+.35*e[5:,j].max(),lambda e:e[0,j]+.25*e[1:5,j].max()+.2*e[5:,j].max()]:
  for k,d,k2,r,e in sorted(rows,key=lambda a:rank(a[-1]))[:8]:print(k,d,k2,r,e[:,j].round(4))
np.save('/tmp/shared_rows.npy',np.array(rows,dtype=object),allow_pickle=True)
