import sys,pickle,itertools as I,numpy as np
sys.path.insert(0,'/holdout_task'); import node_00 as m
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x); f,c=m._calibrate(s['panel_a'],s['telemetry_vec'][-1]); z=m._invert(x,f,c)
print('calibration',f,c,'z range',z.min(0),z.max(0),flush=True)
b=s['panel_b']; fit=m._ols(np.c_[np.ones(len(b['u'])),b['u']],b['v']); bp=lambda a: a@fit[1:]+fit[0]; bi=lambda a:np.linalg.solve(fit[1:].T,(a-fit[0]).T).T
p=s['panel_c']; u=p['u']; print('panel cross-coordinate checks',[(k,len(s[k]['u'])) for k in ['panel_a','panel_b','panel_c']], np.linalg.svd(fit[1:])[1],flush=True)
prod=lambda a:m._products(a)@m._ols(m._products(u),p['v1']); F=lambda q: m._fourier(u,q); q=minimize_scalar(lambda q:np.sum((F(q)@m._ols(F(q),p['v2'])-p['v2'])**2),bounds=(1.35,3.55),method='bounded').x; cos=lambda a:m._fourier(a,q)@m._ols(F(q),p['v2']); print('C frequency',q,flush=True)
coords={'X':x,'Z':z,'B':bp(z),'I':bi(z),'Bc':z@fit[1:],'Ic':np.linalg.solve(fit[1:].T,z.T).T,'XB':bp(x),'XI':bi(x)}
rng=np.random.default_rng(2701); perm=rng.permutation(n); folds=[np.arange(n-20,n),*np.array_split(perm,4)]; pairs=[(np.setdiff1d(np.arange(n),te),te) for te in folds]
def score(w):
 out=[]
 for tr,te in pairs:
  a=(w-w[tr].mean(0))/np.maximum(w[tr].std(0),.1); g=a@a.T/a.shape[1]; d=np.diag(g); D=np.maximum(d[:,None]+d-2*g,0); K=(1+.25*g)**2; L=K+np.exp(-.06*D); eye=np.eye(len(tr)); i=np.ix_(tr,tr); j=np.ix_(te,tr); yp=(K[j]@np.linalg.solve(K[i]+3e-5*eye,y[tr])+L[j]@np.linalg.solve(L[i]+3e-4*eye,y[tr]))/2; e=abs(yp-y[te]); out.append((e[:,0].mean(),e[:,1:].mean()))
 return np.array(out)
features={}; scores=[]
def add(ks):
 r=score(np.column_stack([features[k] for k in ks])); scores.append((np.max(r),np.mean(r),'+'.join(ks),r))
for k,v in coords.items():
 for a,fn in [('p',prod),('c',cos),('l',lambda w:w),('s',lambda w:np.sin(q*w))]: features[a+k]=fn(v)
for a,b in I.product(coords,repeat=2):
 for extras in [(),('lZ',),('lB',),('lI',),('sZ',),('sB',),('sI',)]: add(('p'+a,'c'+b)+extras)
for a,b in I.product(coords,repeat=2):
 add(('l'+a,'c'+b));add(('l'+a,'p'+b))
for a in coords:
 for k in ['l','p','c']: add((k+a,))
print('top worst-fold');
for a,b,k,r in sorted(scores,key=lambda p:p[0])[:22]: print(k,round(a,6),round(b,6),r.round(4).tolist())
print('top public fold');
for a,b,k,r in sorted(scores,key=lambda p:max(p[3][0]))[:12]: print(k,round(a,6),r.round(4).tolist())
for name in ['pZ+cZ','pB+cB','pI+cI','pZ+cZ+lZ','pZ+cZ+lB']:
 print('named',name,next(r for a,b,k,r in scores if k==name))
