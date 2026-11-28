import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from node_00 import _fit,_inverse,_linear,_c1,_c2
from scipy.optimize import minimize_scalar,least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; z=_inverse(s['panel_a'],x); Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y); folds=[np.arange(52,72),*np.array_split(np.random.default_rng(675).permutation(n),5)]
p=s['panel_c']; print('C1 residual',np.std(p['v1']-_c1(p['u'])@_fit(_c1(p['u']),p['v1'])))
I,J=np.triu_indices(4,1); phi=lambda t:np.c_[_linear(t),t*t,t[:,I]*t[:,J]]; print('C1 full',np.round(_fit(phi(p['u']),p['v1']),4));print('C2 coef',np.round(_fit(_c2(p['u']),p['v2']),4))
cc=_fit(_c1(p['u']),p['v1']);dd=_fit(_c2(p['u']),p['v2']); p=s['panel_b'];u,v=p['u'],p['v'];print('U ranges:',[(s[k]['u'].min(),s[k]['u'].max()) for k in ('panel_a','panel_b','panel_c')]);bests=[]
for odd in [lambda t:t**3,lambda t:t*abs(t),lambda t:t,lambda t:t**5]:
 for nm,f in [('tanh',np.tanh),('sin',np.sin),('erf',erf),('rat',lambda t:t/(1+t*t)),('root',lambda t:t/np.sqrt(1+t*t)),('exp',lambda t:np.sign(t)*(-np.expm1(-abs(t))))]:
  for ext in (0,1,3):
   def F(t,w,f=f,odd=odd,ext=ext):return np.c_[np.ones(len(t)),f(w*odd(t)),t**ext] if ext else np.c_[np.ones(len(t)),f(w*odd(t))]
   def err(w):
    A=F(u,w); return np.mean((A@_fit(A,v)-v)**2)
   q=minimize_scalar(err,bounds=(.02,4),method='bounded');bests.append((q.fun**.5,nm,ext,q.x,odd(2),lambda t,f=F,w=q.x,c=_fit(F(u,q.x),v): f(t,w)@c))
bests.sort(key=lambda k:k[0]); print('B best', [k[:5] for k in bests[:15]]);B=bests[0][-1]
def score(g):
 es=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); a=g[tr];d=np.eye(g.shape[1])*.0003;d[0,0]=0;e=abs(g[te]@np.linalg.solve(a.T@a+d,a.T@Y[tr])-Y[te]);es.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(es)
opts=[]
for nm,t in [('raw',x),('inv',z)]:
 c=_c1(t)@cc; d=_c2(t)@dd; b=B(t)
 for cm,C in [('c',c),('d',d),('cd',c+d),('dc',c*d),('center_c',c-cc[0]),('center_d',d-np.mean(d,axis=0))]:
  for a in [-2,-1.5,-1,-.75,-.5,-.3,-.15,.15,.3,.5,.75,1,1.5,2]:
   for h in [t+a*C, b+a*C, C+a*t, C+a*b]:
    e=B(h)
    for ff in [e,np.c_[e,c,d],np.c_[e,b,c,d],np.c_[h,e],np.c_[h,e,b,c,d],np.c_[e,np.tanh(h)]]:
     m=score(np.c_[_linear(t),ff]);opts.append((max(m.max(0)),m[0].tolist(),nm,cm,a,ff.shape[1]))
print('compositions',sorted(opts)[:12])
