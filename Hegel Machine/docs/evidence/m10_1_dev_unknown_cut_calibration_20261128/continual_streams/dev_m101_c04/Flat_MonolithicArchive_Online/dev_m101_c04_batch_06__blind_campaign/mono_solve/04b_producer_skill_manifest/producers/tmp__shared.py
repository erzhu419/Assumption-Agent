import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; z=np.load('/tmp/stages.npz')['b']; t=s['eval_times']; pairs=list(itertools.combinations_with_replacement(range(4),2))
U=np.c_[np.ones(len(z)),z]; Q=np.prod(z[:,pairs],axis=2); F=np.c_[U,Q]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def mae(a):return np.mean(abs(a))
def shared(i):
 u=U[i]; q=Q[i]; w=ls(q-u@ls(u,q),T[i]-u@ls(u,T[i])); _,_,r=np.linalg.svd(w,full_matrices=False); return w@r[0]
def data(i,j):
 w=shared(i); q=Q@w; f=np.c_[U,q]; wt=ls(f[i],T[i]); print('states',mae(f[j]@wt-T[j]),'coeff',np.round(wt,3)) if j[0]==56 else None
 return f,q,wt
I=np.arange(56); J=np.arange(56,72); f,q,wt=data(I,J)
print('y state projection',ls(f[I],y[I]),'mae',mae(f[J]@ls(f[I],y[I])-y[J])); print('state/time residual sv',np.linalg.svd(T[I]-f[I]@wt,compute_uv=False));print('y residual after state shared quadratic',np.round(np.sort(y[I]-f[I]@ls(f[I],y[I])),4))
func={'rat':lambda a:a/(1+a*a),'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'sat':lambda a:a/np.sqrt(1+a*a),'cub':lambda a:a**3,'rat3':lambda a:a**3/(1+a*a)}
def fit(i,j,f,q,wt,name,mode):
 rng=np.random.default_rng(18); base=F if mode=='full' else f; zi=ls(base[i],y[i])[1:5]; dirs=np.r_[zi[None,:],wt[1:5,[0,7]].T,np.eye(4),rng.normal(size=(5,4))]; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
 def bf(k,p):
  a=U[k]@p[:5]; b=base[k]
  if mode=='gated':return np.c_[b,U[k]/(1+a[:,None]**2)]
  if mode=='two':return np.c_[b,func[name](a),func[name](U[k]@p[5:])]
  if mode=='qgate':return np.c_[b,func[name](a+p[5]*q[k])]
  return np.c_[b,func[name](a)]
 def r(p):
  a=bf(i,p);return a@ls(a,y[i])-y[i]
 starts=[np.r_[0,d*a] for d in dirs for a in ([.9,2.] if mode!='two' else [1.6])]; starts=[np.r_[p,0.] for p in starts] if mode=='qgate' else ([np.r_[p,0,dirs[(k+2)%len(dirs)]*1.1] for k,p in enumerate(starts)] if mode=='two' else starts)
 sol=min((least_squares(r,p,max_nfev=90) for p in starts),key=lambda e:e.cost); a=bf(i,sol.x); w=ls(a,y[i]);return mae(sol.fun),mae(bf(j,sol.x)@w-y[j]),sol.x,w
cand=[(n,m) for n in func for m in ('shared','full')]+[('rat',m) for m in ('qgate','gated','two')]+[('tanh','two')]
for n,m in cand:
 a,b,p,w=fit(I,J,f,q,wt,n,m);print(n,m,np.round([a,b],6),np.round(p,4),np.round(w,4) if m=='shared' else '')
