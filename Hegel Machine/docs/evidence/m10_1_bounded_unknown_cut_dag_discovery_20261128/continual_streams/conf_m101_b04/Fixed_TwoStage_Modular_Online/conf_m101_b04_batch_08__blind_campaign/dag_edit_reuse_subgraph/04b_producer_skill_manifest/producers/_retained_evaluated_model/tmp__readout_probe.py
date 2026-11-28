from sink_probe import *
from scipy.optimize import least_squares
from itertools import combinations
folds=[np.arange(52,N)]+list(np.array_split(np.random.default_rng(30).permutation(N),6))

def check(F, penalties=None):
 m=[]
 for v in folds:
  a=np.setdiff1d(np.arange(N),v); X=F[a]; C=np.linalg.lstsq(X,Y[a],rcond=1e-9)[0] if penalties is None else np.linalg.solve(X.T@X+np.diag(penalties),X.T@Y[a]); e=np.abs(F[v]@C-Y[v]);m.append([e[:,0].mean(),e[:,1:].mean()])
 m=np.array(m);return np.r_[m[0],m[1:].mean(0),m[1:].max(0)]
p=s['panel_c']; x=p['u']; B=poly(x,1)
for target in ('v1','v2'):
 V=p[target]
 for fn in (np.tanh,np.sin):
  def residual(q):
   F=np.c_[B,fn(x*q[:4]+q[4:])];return (F@np.linalg.lstsq(F,V,rcond=None)[0]-V).ravel()
  fit=least_squares(residual,np.r_[np.full(4,2.),np.zeros(4)],max_nfev=100,bounds=(np.r_[np.full(4,.15),np.full(4,-3.)],np.r_[np.full(4,8.),np.full(4,3.)])); print('panel',target,fn.__name__,np.round(fit.x,3),np.std(fit.fun))
results=[]
for k in ('xa','xab','z','raw','xa-xab','z-xab'):
 x=d.get(k,s['train']['x']); W=[x] if '-' not in k else [d[t] for t in k.split('-')]; B=poly(x,2)
 for freq in (.6,1,1.4,1.8,2.2,2.6,3.,3.5,4.):
  for mode in ('tanh','sin','tanhcos','pair','products','bounded'):
   cols=[]
   for w in W:
    T=np.tanh(freq*w); P=np.column_stack([w[:,i]+sg*w[:,j] for i,j in combinations(range(4),2) for sg in (-1,1)])
    cols.append({'tanh':T,'sin':np.sin(freq*w),'tanhcos':np.c_[T,np.cos(freq*w)],'pair':np.c_[T,np.tanh(freq*P),np.cos(freq*P)],'products':np.c_[T,np.column_stack([T[:,i]*T[:,j] for i,j in combinations(range(4),2)])],'bounded':np.c_[T,w/(1+freq*w*w),1/(1+freq*w*w)]}[mode])
   F=np.column_stack([B]+cols); std=np.maximum(F.std(0),.1);F=F/std
   for reg in (0.00001,.001,.01,.1):
    m=check(F,np.r_[np.full(B.shape[1],reg*.05),np.full(F.shape[1]-B.shape[1],reg)]);results.append((m, k, freq,mode,reg))
for m,*cfg in sorted(results,key=lambda t:t[0][2])[:20]: print(cfg,np.round(m,5))
for m,*cfg in sorted(results,key=lambda t:t[0][3])[:5]: print('traj',cfg,np.round(m,5))
