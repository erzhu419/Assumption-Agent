exec(open('/tmp/probe.py').read().split('funs=')[0])
from itertools import combinations
from scipy.optimize import minimize_scalar
q=a['_inverse'](s['test']['x'],w); t=s['eval_times']; Y=s['train']['y']; S=s['train']['traj']; T=np.c_[Y,S]
def ph(x,k,p):
 if k=='gauss':return x*(-np.expm1(-p*x*x))
 if k=='rat':return x*(p*x*x)/(1+p*x*x)
 if k=='tanh':return x-np.tanh(p*x)/p
 if k=='sin':return x-np.sin(p*x)/p
 if k=='square':return x*x
 if k=='erf':
  from scipy.special import erf
  return x-erf(p*x)*np.sqrt(np.pi)/(2*p)
def identify(u,v):
 scores=[]
 for k in ['gauss','rat','tanh','sin','square','erf']:
  for lin in [False,True]:
   def design(x,p):return np.c_[aff(x) if lin else np.ones(len(x)),ph(x,k,p)]
   def fit(p):
    F=design(u,p); C=ls(F,v); return (F@C-v),C,F
   p=minimize_scalar(lambda p: np.mean(fit(p)[0]**2),bounds=(.05,5),method='bounded').x
   r,C,F=fit(p); h=np.sum(F*np.linalg.pinv(F).T,axis=1); err=np.sqrt(np.mean((r/(1-h[:,None]))**2)); scores.append((err,k,lin,p,C))
 return sorted(scores,key=lambda p:p[0])
operators={}
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 scores=identify(s[panel]['u'],s[panel][key]); print(panel,key,[(round(e,7),k,b,round(p,6)) for e,k,b,p,c in scores[:8]]); e,k,b,p,c=scores[0]; print('C',c); operators[key]= (lambda x,k=k,b=b,p=p,c=c: np.c_[aff(x) if b else np.ones(len(x)),ph(x,k,p)]@c)
D={'z':z,**{k:f(z) for k,f in operators.items()}}
D.update({'g':ph(z,'gauss',1),'r':ph(z,'rat',1),'g2':ph(z,'gauss',2)})
F={k:aff(v) for k,v in D.items()}
for k,v in D.items():
 F[k+'^2']=poly(v,2); F[k+'sq']=np.c_[aff(v),v*v]
for i,j in combinations(D,2):
 v,w=D[i],D[j]; F[i+'+'+j]=np.c_[aff(v),w]; F[i+'*'+j]=np.c_[aff(v),w,v*w]
for k,v in list(D.items())[1:4]:
 ix,jx=np.triu_indices(4,1); F[k+'+zoff']=np.c_[aff(v),z,z[:,ix]*z[:,jx]]; F[k+'+zquad']=np.c_[aff(v),poly(z,2)[:,1:]]; F[k+'off']=np.c_[aff(v),v[:,ix]*v[:,jx]]
rng=np.random.default_rng(2026); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(8)]; stats=[]
for k,f in F.items():
 errs=[]
 for ids in splits:
  tr,va=ids[:52],ids[52:]; e=abs(f[va]@ls(f[tr],T[tr])-T[va]);errs.append([e[:,0].mean(),e[:,1:].mean()])
 stats.append((k,np.array(errs)))
for metric in [0,1]:
 print('TARGET',metric)
 for k,e in sorted(stats,key=lambda a:a[1][:,metric].max())[:18]:print(k,'official',e[0], 'worst',e.max(0),'mean',e.mean(0))
