import sys,pickle,itertools as it,numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
sys.path.insert(0,'/holdout_task');import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']);z=m._panel_a(s,tr['x']); b=m._panel_b(s,z)
D=m._design; fit=m._fit
pc=s['panel_c']; u=pc['u']; a,c=np.triu_indices(4,1)
models=[]
for key in ['v1','v2']:
 v=pc[key]; opts=[]
 for fname,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('erf',m.erf)]:
  for inter in range(3):
   def des(x,q):
    g=f(q*x); return D(g if inter==0 else np.c_[x,g] if inter==1 else np.c_[x,g,x[:,a]*x[:,c],x**2])
   def err(q):
    F=des(u,q);return np.mean((F@fit(F,v)-v)**2)
   r=min([minimize_scalar(err,bounds=(q,q+.6),method='bounded') for q in np.arange(.25,5,.6)],key=lambda r:r.fun)
   co=fit(des(u,r.x),v);opts.append((r.fun,des(z,r.x)@co,des(b,r.x)@co)); print(key,fname,inter,round(r.x,4),round(r.fun**.5,6),co.round(3).tolist() if r.fun<2e-7 and inter==0 else '')
 models.append(min(opts,key=lambda o:o[0]))
h,g=models[0][1],models[1][1]
_,_,vh=np.linalg.svd(tr['traj'],full_matrices=False); U=vh[:2]; y=np.c_[tr['y'],tr['traj']@U.T]; ystd=y.std(0); Y=(y-y.mean(0))/ystd
rng=np.random.default_rng(9); idx=np.array([np.arange(n),*[rng.permutation(n) for _ in range(6)]]); splits=[(r[:52],r[52:]) for r in idx]
def score(pred,ix):
 e=(pred-Y[ix])*ystd; return np.array([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@U))])
def norm(F):return (F-F.mean(0))/np.maximum(F.std(0),1e-7)
def cv(K,lam):
 return np.array([score((K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i]-Y[i].mean(0),assume_a='pos')+Y[i].mean(0)),j) for i,j in splits])
def ker(F,typ):
 L=F@F.T/F.shape[1]; d=np.maximum(0,np.diag(L)[:,None]+np.diag(L)-2*L)
 if typ==0:return (1+L)**2
 if typ==1:return (1+L)**3
 return (1+L)**2+np.exp(-typ*d)
qdict={'z':z,'b':b,'h':h,'g':g,'H':models[0][2],'G':models[1][2]}; out=[]
for names in ['z','b','zb','bh','bg','zh','zg','zbh','zbg','bhg','zbHG','zbhg']:
 F=norm(np.c_[tuple(qdict[k] for k in names)]) if False else norm(np.column_stack([qdict[k] for k in names]))
 for scale in [.4,1.,2.5]:
  for typ in [0,1,.25,1,4]:
   for lam in [.0001,.004,.04]:
    v=cv(ker(F*scale,typ),lam);out.append((v.mean(0).max(),names,scale,typ,lam,v[0],v.max(0)))
print('kernels');print(*sorted(out,key=lambda r:r[0])[:10],sep='\n')
# Search compact additive and tensor target basis.
lib={}
for label,x in list(qdict.items())+[(f'{k}{l}{w:+g}',qdict[k]+w*qdict[l]) for k,l in [('b','z'),('b','h'),('b','g')] for w in [-1,-.5,-.3,.3,.5,1]]:
 for j,t in enumerate(x.T):
  lib[f'{label}:{j}']=t
  for p in [2,3]:lib[f'{label}:{j}^{p}']=t**p
  for q in [.65,1.2,1.8,2.15,2.6,3.1,4.]:
   for f in [np.sin,np.cos,np.tanh,np.arctan]:lib[f'{f.__name__}({q}*{label}:{j})']=f(q*t)
X=np.column_stack(list(qdict.values())); lab=[f'{k}:{j}' for k in qdict for j in range(4)]
for j,k in it.combinations(range(X.shape[1]),2):lib[lab[j]+'*'+lab[k]]=X[:,j]*X[:,k]
F=norm(np.column_stack(list(lib.values()))); names=list(lib)
def greedy(i,j,kmax):
 fi=F[i]-F[i].mean(0); fj=F[j]-F[i].mean(0); yn=Y[i]-Y[i].mean(0); R=yn.copy(); sel=[]; scales=np.linalg.norm(fi,axis=0); res=[]
 for k in range(kmax):
  p=np.argmax(np.linalg.norm(fi.T@R,axis=1)/np.maximum(scales,1e-10));sel.append(p); coef=fit(fi[:,sel],yn); R=yn-fi[:,sel]@coef;res.append(score(fj[:,sel]@coef+Y[i].mean(0),j))
 return np.array(res),sel
v=np.array([greedy(i,j,30)[0] for i,j in splits]);print('greedy pooled',[(k,v[:,k-1].mean(0),v[0,k-1]) for k in [5,10,15,20,25,30]]); _,sel=greedy(splits[0][0],splits[0][1],22);print('terms',[names[k] for k in sel]);
for inds in [np.arange(F.shape[1]),[i for i,k in enumerate(names) if '+' not in k and '-' not in k],[i for i,k in enumerate(names) if not any(a in k for a in 'hHgG')]]:
 K=F[:,inds]@F[:,inds].T/len(inds); print('dict',len(inds),[(lam,cv(K+1,lam).mean(0),cv(K+1,lam)[0]) for lam in [.00001,.0001,.001,.01,.05,.2]])
