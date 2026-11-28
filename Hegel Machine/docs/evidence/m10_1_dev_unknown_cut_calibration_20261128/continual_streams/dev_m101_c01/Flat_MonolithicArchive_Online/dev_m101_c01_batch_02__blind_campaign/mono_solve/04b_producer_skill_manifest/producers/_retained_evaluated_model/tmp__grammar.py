from operators import *
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from itertools import combinations
x=S['train']['x']; y=S['train']['y']; z=S['train']['traj']; t=S['eval_times']; n=len(x)
rng=np.random.default_rng(451); splits=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in [rng.permutation(n) for _ in range(12)]]
base=np.c_[affine(x),powers(x,2)]; P=base@base.T
H=np.c_[np.ones(n), x]; D=cdist(x,x,'sqeuclidean'); K0=np.exp(-D/(2*1.1**2)); K1=np.exp(-D/(2*1.6**2))
A=[ls(modes(t,z[tr]),z[tr].T).T for tr,te in splits]; M=[modes(t,z[tr]) for tr,te in splits]
models={};
for name,w in coords(x,S).items():
 if name not in ('raw','A','Ai','B','BA','BAi','Bi','BiA','BiAi','AB','AiB'): continue
 print(name,'degree3+4',np.round(ls(np.c_[base,w**3,w**4],np.c_[y,ls(modes(t,z),z.T).T])[-8:],3).T)
 d=(w[:,None,:]-w[None,:,:])**2
 # Separate pure-axis curvature, pair interactions, and full-dimensional remainder.
 for length in (.45,.7,1.,1.5):
  un=np.exp(-d/(2*length**2)); un=un-un.mean(0)[None,:,:]-un.mean(1)[:,None,:]+un.mean((0,1)); U=un.sum(2); V=sum(un[:,:,i]*un[:,:,j] for i,j in combinations(range(4),2)); W=sum(un[:,:,i]*un[:,:,j]*un[:,:,k] for i,j,k in combinations(range(4),3))
  for label,extra in [('add',U),('pair',U+V),('weakpair',U+.15*V),('three',U+V+.5*W)]:
   models[name,label,length]=extra
 for label,f in [('diag',np.c_[w**3,w**4]),('diag5',np.c_[w**3,w**4,w**5]),('diag+cross',np.c_[w**3,w**4,powers(w,3)*.35]),('triple',np.column_stack([np.prod(w[:,p],axis=1) for p in combinations(range(4),3)])),('p4',np.c_[powers(w,3),powers(w,4)*.5]),('radial',np.c_[w**3,w**4,np.sum(w*w,axis=1)**2])]:
  models[name,label,0]=f@f.T
res=[]
for key,U in models.items():
 for weight in (.03,.2,1.):
  for mix,K in [('pure',0),('s',.1*K0),('l',K1)]:
   P0=P+weight*U+K
   for nr in (.0003,.002,.01):
    G=P0+nr*np.eye(n); es=[]
    for j,(tr,te) in enumerate(splits):
     pred=G[np.ix_(te,tr)]@solve(G[np.ix_(tr,tr)],np.c_[y[tr],A[j]],assume_a='pos'); er=np.abs(pred[:,1:]@M[j].T-z[te]); es.append((np.mean(abs(pred[:,0]-y[te])),er.mean(),er[:,-1].mean()))
    res.append((key+(weight,mix,nr),np.array(es)))
for j in range(3):
 print('metric',j)
 for k,e in sorted(res,key=lambda p:p[1][:,j].mean()+.35*p[1][:,j].max())[:8]: print(k,np.round(e[0],5),np.round(e.mean(0),5),np.round(e.max(0),5))
