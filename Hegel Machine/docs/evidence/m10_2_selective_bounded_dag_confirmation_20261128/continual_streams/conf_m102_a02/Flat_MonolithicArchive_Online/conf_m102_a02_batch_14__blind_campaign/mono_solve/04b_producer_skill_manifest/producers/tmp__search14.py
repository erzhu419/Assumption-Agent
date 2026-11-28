from graph14 import *
from scipy.optimize import differential_evolution
B=S['panel_b']; C=S['panel_c']; tr=S['train']; n=52; Y=np.c_[tr['y'],tr['traj']]; z=m._inverse(tr['x'],calibrate(S)); i,j=np.triu_indices(4)
def fitop(u,v,terms,linear=True):
 def D(x,p):return np.c_[np.ones(len(x)),*([x] if linear else []),*[F[k](x,a) for k,a in zip(terms,p)]]
 def obj(p):
  d=D(u,p);return np.mean((d@ls(d,v)-v)**2)
 opt=differential_evolution(obj,[(.03,6)]*len(terms),seed=0,popsize=8,maxiter=60,tol=1e-7); w=ls(D(u,opt.x),v)
 return lambda x:D(x,opt.x)@w,opt.x,opt.fun**.5
F.update({'pwrat':lambda x,p:x/(1+abs(x)**p),'tanh3':lambda x,p:np.tanh(p*x**3),'erf':lambda x,p:__import__('scipy').special.erf(p*x),'gauss3':lambda x,p:x*np.exp(-p*abs(x)**3),'absrat':lambda x,p:x/(1+p*abs(x)**3),'atan3':lambda x,p:np.arctan(p*x**3),'sin3':lambda x,p:np.sin(p*x**3)})
opts=[]
for terms in [(a,) for a in F if a!='sinh']+list(itertools.combinations(['rat','gauss','tanh','soft','sin','gauss3','tanh3','pwrat','absrat'],2)):
 op,p,e=fitop(B['u'],B['v'],terms);opts.append((e,terms,p,op))
print('B ranking',[(e,k,p) for e,k,p,_ in sorted(opts,key=lambda a:a[0])[:14]],flush=True)
bo=sorted(opts,key=lambda a:a[0])[0][-1]; b0=panel(B['u'],B['v'],['gauss'])[0]; co=panel(C['u'],C['v1'],['rat','soft'])[0]; do=panel(C['u'],C['v2'],['exp'])[0]; P={'b':bo,'g':b0,'c':co,'d':do}; L={'z':z,'x':tr['x'],**{k:op(z) for k,op in P.items()}}
for a,b in itertools.product(P,P):L[a+b]=P[b](L[a])
for a,b in itertools.product('bc','bcd'):
 for c in 'bcd':L[a+b+c]=P[c](L[a+b])
for k,a in list(L.items()):
 if k not in ('z','x'): L['z+'+k]=z+a; L['z-'+k]=z-a
out=[]
for keys in [(k,) for k in L]+list(itertools.combinations([k for k in L if '+' not in k and '-' not in k],2)):
 a=np.column_stack([L[k] for k in keys]); f=feats(a); f=(f-np.r_[0,f[:n,1:].mean(0)])/np.r_[1,np.maximum(f[:n,1:].std(0),.015)];
 for la in [0,.003,.05,.3]:
  w=ls(np.r_[f[:n],np.sqrt(la)*np.eye(f.shape[1])],np.r_[Y[:n],np.zeros((f.shape[1],9))]); E=abs(f[n:]@w-Y[n:]); Et=abs(f[:n]@w-Y[:n]);out.append((E[:,0].mean(),E[:,1:].mean(),Et[:,0].mean(),Et[:,1:].mean(),keys,la))
print('scalar',*sorted(out)[:20],sep='\n');print('traj',*sorted(out,key=lambda a:a[1])[:20],sep='\n'); print('fit',*sorted(out,key=lambda a:a[2]+a[3])[:12],sep='\n'); print('times',S['eval_times'])
