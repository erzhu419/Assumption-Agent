import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
fit=m._fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; S,F,G,X=m._design(s); Y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; n=len(Y); k=F.shape[1]

def cols(f,g,b):
 return np.column_stack(((f[:,None,:]*b[0][None,:,None]).reshape(-1,f.shape[1]),(g[:,None,:]*b[1][None,:,None]).reshape(-1,g.shape[1])))
def profiles(kind,w):
 e=-np.expm1(-w[0]*t)
 funcs={'exp':lambda h:-np.expm1(-h*t),'log':lambda h:np.log1p(h*t),'rat':lambda h:t/(1+h*t),'sin':lambda h:np.sin(h*t)}
 return np.array([e,funcs[kind](w[1])])
def pred(a,b,kind):
 f,g=F[a],G[a]; z=T[a]; A=np.c_[f,g]; coef=fit(A,z)
 if kind=='plain':return np.c_[F[b],G[b]]@coef, np.mean((A@coef-z)**2)**.5, []
 if kind=='als':
  basis=np.stack([np.linalg.svd(q,full_matrices=False)[2][0] for q in (coef[:k],coef[k:])]);
  for _ in range(30):
   w=fit(cols(f,g,basis),z.ravel()); basis=fit(np.c_[f@w[:k],g@w[k:]],z); basis/=np.linalg.norm(basis,axis=1)[:,None]
  p=[]
 else:
  def fun(w):
   D=cols(f,g,profiles(kind,w));return D@fit(D,z.ravel())-z.ravel()
  p=least_squares(fun,[1.8,.45],bounds=([.05,.001],[6.,2.]),gtol=1e-10).x; basis=profiles(kind,p)
 D=cols(f,g,basis); w=fit(D,z.ravel());return (cols(F[b],G[b],basis)@w).reshape(len(b),-1),np.mean((D@w-z.ravel())**2)**.5,p
splits=[('52/20',np.arange(52),np.arange(52,n)),('56/16',np.arange(56),np.arange(56,n))]
for i in range(12):
 p=np.random.default_rng(i).permutation(n);splits.append(('random',p[:56],p[56:]))
for j in range(4):
 p=np.argsort(X[:n,j]);splits.extend([('tail',p[:56],p[56:]),('tail',p[16:],p[:16])]);
for kind in ('plain','als','exp','log','rat','sin'):
 rows=[]
 for label,a,b in splits:
  scalar=S[b]@fit(S[a],Y[a]); traj,rms,p=pred(a,b,kind); e=abs(traj-T[b]); row=[abs(scalar-Y[b]).mean(),e.mean(),e.mean(0).max(),e[:,-1].mean()];rows.append(row)
  if label=='52/20':print(kind,'52/20',np.round(row,6),'train rms',round(rms,6),'rates',p)
 rows=np.array(rows); print(kind,'random max',np.round(rows[2:14].max(0),6),'tail max',np.round(rows[14:].max(0),6),'mean',np.round(rows.mean(0),6))
