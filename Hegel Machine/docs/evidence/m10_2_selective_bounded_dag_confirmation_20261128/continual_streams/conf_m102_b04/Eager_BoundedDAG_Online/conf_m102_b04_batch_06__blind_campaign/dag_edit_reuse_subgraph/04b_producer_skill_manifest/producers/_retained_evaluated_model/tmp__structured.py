exec(open('/tmp/probe.py').read().split('for name in')[0])
from scipy.optimize import minimize_scalar
x=np.load('/tmp/inverse_probe.npz')['latent']; y=s['train']['y']; z=s['train']['traj']; Y=np.c_[y,z]; n=len(x); idx=np.arange(n); rng=np.random.default_rng(23); folds=[(idx[:52],idx[52:])]+[(np.setdiff1d(idx,j),j) for j in np.array_split(rng.permutation(n),4)]
def fn(q,a,k):
 if k=='sqrt':return q/np.sqrt(1+a*q*q)
 if k=='cube-rational': return q**3/(1+a*q*q)
 if k=='cube-sqrt':return (q/np.sqrt(1+a*q*q))**3
 if k=='cube-tanh':return (np.tanh(np.sqrt(a)*q)/np.sqrt(a))**3
 if k=='cube-sin':return (np.sin(np.sqrt(a)*q)/np.sqrt(a))**3
 if k=='cube-atan':return (np.arctan(np.sqrt(a)*q)/np.sqrt(a))**3
 if k=='cube-tanh3':return np.tanh(a*q**3)/a
 if k=='cube-rat6':return q**3/np.sqrt(1+a*q**6)
 if k=='cube-rat3':return q**3/(1+a*abs(q)**3)
def feats(a,k,b):
 q=fn(x,a,k); ones=np.ones((n,1));
 if k=='sqrt':return pol(q,3)
 if b==0:return np.c_[ones,q]
 if b==1:return np.c_[ones,q,x]
 if b==2:return np.c_[pol(x,2),q]
 if b==3:return np.c_[ones,q,q*q,np.column_stack([q[:,i]*q[:,j] for i in range(4) for j in range(i)])]
 if b==4:return np.c_[ones,q,np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i)])]
 if b==5:return np.c_[ones,q,x*x]
for k in ['sqrt','cube-rational','cube-sqrt','cube-tanh','cube-sin','cube-atan','cube-tanh3','cube-rat6','cube-rat3']:
 for b in range(1 if k=='sqrt' else 6):
  ers=[]; params=[]
  for i,j in folds:
   iF=lambda a: feats(a,k,b)[i];
   def loss(a):
    f=iF(a); e=f@np.linalg.lstsq(f,Y[i],rcond=None)[0]-Y[i]; return (e*e).mean()
   opt=minimize_scalar(loss,bounds=(.001,1.7),method='bounded'); f=feats(opt.x,k,b); e=f[j]@np.linalg.lstsq(f[i],Y[i],rcond=None)[0]-Y[j]; ers.append([abs(e[:,0]).mean(),abs(e[:,1:]).mean()]); params.append(opt.x)
  if np.max(ers)<.042:
   print(k,b,np.round(params,3),np.round(ers,4).tolist())
