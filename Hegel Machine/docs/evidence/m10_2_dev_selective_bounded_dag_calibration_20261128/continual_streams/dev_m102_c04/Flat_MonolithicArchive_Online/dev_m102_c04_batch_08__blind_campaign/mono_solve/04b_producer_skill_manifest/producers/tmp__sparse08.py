from graph08 import *
from scipy.linalg import svd
from scipy.optimize import minimize_scalar
D=setup(s)(s['train']['x']); z=D['z']; c=s['panel_c']; u=c['u']; Q=lambda x:np.c_[x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]]; C=lambda x:np.cos(2.4*x); W=ls(aff(Q(u)),c['v1']); V=ls(aff(C(u)),c['v2']); p=lambda x:aff(Q(x))@W; r=lambda x:aff(C(x))@V
D.update({'q':Q(z),'c':C(z),'qb':Q(D['b']),'cb':C(D['b']),'rp':r(D['p']),'pr':p(D['r']),'qr':Q(D['r']),'cp':C(D['p'])})
T=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; N=52
_,sig,H=svd(T[:N],full_matrices=False); Y=np.c_[y,T@H[:3].T]; Y=Y/Y[:N].std(axis=0); print('time basis',t,H[:3]); print('target correlations',np.corrcoef(Y.T))
base=aff(z,Q(z),C(z));
cols={}; blocks={}
def put(n,A):
 A=np.asarray(A); A=A[:,None] if A.ndim==1 else A; blocks[n]=A
 for j,a in enumerate(A.T):cols[f'{n}[{j}]']=a
for n,x in D.items(): put(n,x)
for n in ['z','b']:
 x=D[n];put(n+'poly2',quad(x));put(n+'poly3',np.column_stack([np.prod(x[:,i],axis=1) for i in itertools.combinations_with_replacement(range(4),3)]));put(n+'poly4',np.column_stack([np.prod(x[:,i],axis=1) for i in itertools.combinations_with_replacement(range(4),4)]))
for n,m in itertools.combinations_with_replacement(['z','b','p','r','P','R','q','c','qb','cb'],2):
 x,v=D[n],D[m];put(n+'*'+m,np.einsum('ni,nj->nij',x,v).reshape(-1,16));put(n+'.'+m,np.sum(x*v,axis=1));put(n+'had'+m,x*v)
for n,m in itertools.product(['p','r','P','R','z','b'],repeat=2):
 x,v=D[n],D[m];
 for k in (.1,.22,.5,1,1.25,2.4):
  put(f'{n}/{m}sq{k}',x/(1+k*v*v)); put(f'{n}tanh{m}{k}',x*np.tanh(k*v)); put(f'{n}sig{m}{k}',x/(1+np.exp(-k*v)))
 for k in (.22,1,1.25,2.4): put(f'tanh{k}{n}+{m}',np.tanh(k*(x+v)));put(f'cos{k}{n}-{m}',np.cos(k*(x-v)))
for n in ['p','r','P','R','q','c','z','b']:
 x=D[n]
 for k in (.22,1,1.25,2.4): put(f'tanh{k}{n}',np.tanh(k*x));put(f'cos{k}{n}',np.cos(k*x))
keys=list(cols); A=np.column_stack(list(cols.values()));
def report(B):
 W=ls(B[:N],Y[:N]); e=Y-B@W; return tuple(np.round(np.mean(abs(e[N:]),0),4)),tuple(np.round(np.std(e[:N],0),4))
print('base report',report(base)); print('single block best',sorted((report(np.c_[base,a]),n) for n,a in blocks.items())[:16]);
for basis in [aff(z),base,aff(z,quad(z),C(z))]:
 for idx in [0,1,2,3,-1]:
  B=basis.copy(); out=[]; target=Y if idx<0 else Y[:,idx:idx+1]
  for j in range(12):
   E=target[:N]-B[:N]@ls(B[:N],target[:N]); F=A[:N]-B[:N]@ls(B[:N],A[:N]); den=np.linalg.norm(F,axis=0); gain=np.linalg.norm(F.T@E,axis=1)/np.maximum(den,1e-12); gain[den<1e-7]=0; k=np.argmax(gain); B=np.c_[B,A[:,k]]; out.append((keys[k],report(B)))
  print('greedy',basis.shape[1],idx,out)
