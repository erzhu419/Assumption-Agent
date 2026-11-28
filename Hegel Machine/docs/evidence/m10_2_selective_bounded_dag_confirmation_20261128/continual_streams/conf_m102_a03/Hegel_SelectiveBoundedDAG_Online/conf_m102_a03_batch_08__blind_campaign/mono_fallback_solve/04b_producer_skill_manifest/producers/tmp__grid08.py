import numpy as np,pickle
from scipy.linalg import cho_factor,cho_solve
from itertools import combinations_with_replacement as cwr
D=np.load('/tmp/feat08.npz'); E=np.load('/tmp/enc08.npz'); z,b,B,y,t=[D[k] for k in ['z','b','B','y','t']]; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('times',s['eval_times'],'summary',s['train']['summary']); print(np.c_[y,t][:14].round(3)); print('corr',np.corrcoef(np.c_[y,t].T)[0]);
rng=np.random.default_rng(555); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(6)]; std=lambda X:(X-X.mean(0))/np.maximum(X.std(0),1e-8); metrics=[]
def scan(name,K):
 for a in [.0002,.001,.005,.025,.10,.4]:
  err=[]; ter=[]
  for p in splits:
   i,j=p[:52],p[52:]; V=np.c_[y,t]; m=V[i].mean(0); r=K[np.ix_(j,i)]@cho_solve(cho_factor(K[np.ix_(i,i)]+a*np.eye(len(i))),V[i]-m)+m-V[j]; err.append(np.mean(abs(r[:,0]))); ter.append(np.mean(abs(r[:,1:])))
  metrics.append((np.mean(err),max(err),err[0],a,name,np.mean(ter),max(ter),ter[0]))
A=std(z)
for name,X in {'a':z,'b':b,'B':B,**{k:E[k] for k in E.files if k!='z'}, **{f'sin{f}':np.sin(f*z) for f in [1.7,2.3,2.8,3.1,3.5,4.2]}, **{f'coord{f:g},{g:g}':z-g*np.sin(f*z) for f in [2.,2.4,2.7,2.9,3.1,3.4] for g in [.25,.4,.55]}, **{f'odd{f:g},{g:g}':z-g*np.tanh(f*z) for f in [1.5,2,2.5] for g in [.55,.75,1.]}}.items():
 X=std(X); G=X@X.T/4; GA=A@A.T/4; R=(X[:,None,:]-X[None,:,:])**2; RA=(A[:,None,:]-A[None,:,:])**2
 for h in [0.,.1,.4,1.]:
  Q=(G+h*GA)/(1+h)
  for d in [2,3,4]:scan((name,'poly',h,d),(1+Q)**d)
  for L in [.7,2,6,15]:
   r=np.exp(-(R+h*RA)/(L*(1+h))); m=r.sum(2); wt=[m/4,(m*m-(r*r).sum(2))/12];
   scan((name,'joint',h,L),(1+Q)**2+3*np.exp(-(R+h*RA).mean(2)/(L*(1+h))))
   scan((name,'add',h,L),(1+Q)**2+wt[0]+wt[1]);
print('BEST norm by mean');print(*sorted(metrics)[:22],sep='\n'); print('BEST by split'); print(*sorted(metrics,key=lambda r:r[2])[:14],sep='\n'); print('BEST state');print(*sorted(metrics,key=lambda r:r[5])[:16],sep='\n')
