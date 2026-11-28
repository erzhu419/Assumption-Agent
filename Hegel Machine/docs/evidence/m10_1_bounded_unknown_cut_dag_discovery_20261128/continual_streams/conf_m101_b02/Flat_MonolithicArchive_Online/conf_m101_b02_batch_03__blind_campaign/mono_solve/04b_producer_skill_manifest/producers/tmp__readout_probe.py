import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; Y=np.c_[y,z]
def D(*a): return np.column_stack((np.ones(len(a[0])),*a))
def L(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def p2(a): return np.array([a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(a.shape[1]),2)]).T
def cross(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
p=s['panel_b']; B=L(D(p['u']),p['v']); b=lambda a:D(a)@B
p=s['panel_c']; h=lambda a:-np.sign(a)*np.expm1(-1.8*abs(a)); j=lambda a:D(a,a*a/np.sqrt(1+.39*a*a)); C=L(D(h(p['u'])),p['v1']); E=L(j(p['u']),p['v2']); c=lambda a:D(h(a))@C; d=lambda a:j(a)@E
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.argsort(np.linalg.norm(D(u)@L(D(u),v)-v,axis=1))[:45]; u,v=u[keep],v[keep]; qa=lambda a:D(a,p2(a),a**3); A=L(qa(u),v); print('A poly residual',np.sqrt(np.mean((qa(u)@A-v)**2))); a=lambda t:qa(t)@A
nodes={'x':x,'b':b(x),'a':a(x),'ab':b(a(x)),'ba':a(b(x))}; G={}
for name,f in nodes.items():
 g,k=c(f),d(f)
 G.update({name+'/f':D(f),name+'/quad':D(f,p2(f)),name+'/fc':D(f,g),name+'/fd':D(f,k),name+'/fcd':D(f,g,k),name+'/dot':D(f,g,k,(f*g).sum(1),(f*k).sum(1),(g*k).sum(1)),name+'/FC':D(f,g,cross(f,g)),name+'/FD':D(f,k,cross(f,k)),name+'/CD':D(g,k,cross(g,k)),name+'/all':D(f,g,k,cross(f,g),cross(f,k),cross(g,k)),name+'/fcdq':D(f,g,k,p2(f)),name+'/fcdfull':D(f,g,k,p2(np.c_[f,g,k]))})
rng=np.random.default_rng(882); ix=np.arange(72); folds=[(ix[:52],ix[52:]),(ix[:56],ix[56:])]+[(q[:52],q[52:]) for q in [rng.permutation(72) for _ in range(5)]]
R=[]
for name,g in G.items():
 for lam in (1e-7,1e-4,.003,.03,.3):
  es=[]
  for tr,te in folds:
   f=g[tr]; diag=np.eye(g.shape[1])*lam;diag[0,0]=0
   pt=g[te]@np.linalg.solve(f.T@f+diag,f.T@Y[tr]); er=abs(Y[te]-pt);es.append([er[:,0].mean(),er[:,1:].mean()])
  es=np.array(es); R.append((name,lam,*es[0],*es[1],*es.max(0),*es.mean(0)))
for i in (2,3,6,7):
 print('SORT',i); print(*[(q[0],q[1],*np.round(q[2:],5)) for q in sorted(R,key=lambda q:q[i])[:10]],sep='\n')
