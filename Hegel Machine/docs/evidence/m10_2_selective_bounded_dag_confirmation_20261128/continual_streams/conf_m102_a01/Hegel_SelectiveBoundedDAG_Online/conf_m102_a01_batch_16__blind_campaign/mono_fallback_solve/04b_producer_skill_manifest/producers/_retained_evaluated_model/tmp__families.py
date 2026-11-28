import numpy as np,pickle,importlib.util
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=S['train']['x']; Y=S['train']['y']; T=S['train']['traj']; n=len(X)
p='/tmp/hegel_continual_archive/conf_m102_a01_batch_03_solver.py'; spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
A=lambda x: np.column_stack([np.ones(len(x)),x]); mae=lambda x: np.mean(abs(x),axis=0)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
from scipy.optimize import minimize_scalar
from itertools import combinations
p=S['panel_c']; u=p['u']; V=p['v1']; W=p['v2']; print('scalars',S['telemetry_vec'],S['train']['summary']);
for name,fn in [('tanh',lambda x,a:np.tanh(a*x)),('sin',lambda x,a:np.sin(a*x)),('rat',lambda x,a:x/(1+a*x*x)),('atan',lambda x,a:np.arctan(a*x)),('asinh',lambda x,a:np.arcsinh(a*x)),('quad',lambda x,a:x+a*x*x),('quad-cubic',lambda x,a:x+.45*x*x+a*x**3)]:
 for m,target in [(V,'v1'),(W,'v2')]:
  def fit(a,n=24,extra=False):
   d=A(fn(u,a)); d=np.c_[d,u] if extra else d
   e=d@np.linalg.lstsq(d[:n],m[:n],rcond=None)[0]-m
   return np.sqrt(np.mean(e[:n]**2)), np.sqrt(np.mean(e[n:]**2)) if n<24 else 0
  for extra in ([False,True] if name in ('tanh','sin','rat','atan','asinh') else [False]):
   f=minimize_scalar(lambda a:fit(a,18,extra)[0],bounds=(.005,2.5),method='bounded'); print('panel',target,name,extra,round(f.x,6),fit(f.x,18,extra))
z,h=old._coordinates(S,X); p=S['panel_b']; B=np.linalg.lstsq(A(p['u']),p['v'],rcond=None)[0]; C={'x':X,'z':z,'bz':A(z)@B,'bx':A(X)@B,'h':h,'ih':np.linalg.solve(B[1:].T,(X-B[0]).T).T}; p=S['panel_c']; bv=np.linalg.lstsq(A(np.tanh(p['u'])),V,rcond=None)[0]; bw=np.linalg.lstsq(A(p['u']+.45*p['u']**2),W,rcond=None)[0]
for k,g in list(C.items()): C['v'+k]=A(np.tanh(g))@bv; C['w'+k]=A(g+.45*g*g)@bw
Folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(28).permutation(72),4)]; targets=np.c_[Y,T@(-np.expm1(-.8*S['eval_times']))/sum(np.expm1(-.8*S['eval_times'])**2)]
def score(D,la):
 vals=[]; D=np.asarray(D); D/=np.maximum(np.sqrt(np.mean(D*D,axis=0)),1e-12); P=np.r_[np.full(5,1e-8),np.broadcast_to(la,(D.shape[1]-5,))];
 for i,j in Folds:
  d=D[i]; fit=np.linalg.solve(d.T@d+np.diag(P),d.T@targets[i]); vals.append(np.mean(abs(D[j]@fit-targets[j]),axis=0))
 vals=np.array(vals);return np.r_[vals[0],vals[1:].mean(0),vals[1:].max(0)]
rows=[]
for k,g in C.items():
 q=lambda a: np.column_stack([np.prod(a[:,i],axis=1) for i in combinations_with_replacement(range(4),2)]); r=np.column_stack([np.prod(g[:,i],axis=1) for i in combinations_with_replacement(range(4),3)]);
 for a in [.35,.7,1.,1.4,2.]:
  for kind,f in [('tanh',np.tanh(a*g)),('sin',np.sin(a*g)),('rat',g/(1+a*g*g)),('gauss',np.exp(-a*g*g)),('atan',np.arctan(a*g))]:
   for f1,d in [('qf',np.c_[A(X),q(g),f]),('fq',np.c_[A(X),f,q(f)]),('qfq',np.c_[A(X),q(g),f,q(f)]),('diag',np.c_[A(X),g*g,f]),('cross',np.c_[A(X),g*g,f,g*f,q(f)])]:
    for l in [.001,.03,.25,1.]: rows.append((score(d.copy(),l), (k,a,kind,f1,l)))
 for l in [.001,.03,.25,1.]:
  for k1,d in [('q',q(g)),('r',np.c_[q(g),r]),('diag3',np.c_[q(g),g**3]),('diag5',np.c_[q(g),g**3,g**4,g**5])]: rows.append((score(np.c_[A(X),d], l),(k,k1,l)))
for idx in range(4):
 print('RESULT',idx)
 for v,hyp in sorted(rows,key=lambda v:v[0][idx])[:13]: print(*np.round(v,5),hyp)
