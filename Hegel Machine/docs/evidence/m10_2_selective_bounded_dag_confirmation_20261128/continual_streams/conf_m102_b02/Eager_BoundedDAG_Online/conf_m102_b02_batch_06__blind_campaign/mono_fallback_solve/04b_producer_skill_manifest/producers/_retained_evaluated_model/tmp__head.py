from probe import *
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations
import importlib.util
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py'); old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
tr=s['train'];z,b=old.parts(s);z=z[:72];b=b[:72];p=s['panel_c'];u=p['u'];f=lambda a:D(a*np.roll(a,-1,axis=1));c=f(z)@fit(f(u),p['v1']);f=lambda a:D(np.c_[a,np.cos(1.25*a)]);d=f(z)@fit(f(u),p['v2']);
T=tr['traj']; y=tr['y']; t=s['eval_times']; U,S,V=np.linalg.svd(T,full_matrices=False); P=np.c_[y,U[:,:3]*S[:3]]; print('head corr',np.corrcoef(P.T));print('range std',[(q.min().round(3),q.max().round(3),q.std().round(3)) for q in [z,b,c,d,y,T]]);print('rels y',fit(D(T),y),np.std(D(T)@fit(D(T),y)-y));print('time modes',V[:3]);
rng=np.random.default_rng(93); splits=[(np.arange(52),np.arange(52,72))]+[(q[:52],q[52:]) for q in (rng.permutation(72) for _ in range(7))]; T2=(T@V[:3].T)@V[:3]; Y=np.c_[y,T2];
opts=[]
def test(K,name,ridges=(1e-5,.0003,.003,.03,.15)):
 for r in ridges:
  er=[]
  for a,i in splits:
   w=solve(K[np.ix_(a,a)]+r*np.eye(len(a)),Y[a],assume_a='pos'); err=abs(K[np.ix_(i,a)]@w-np.c_[y,T][i]); er.append([err[:,0].mean(),err[:,1:].mean()]);
  E=np.array(er);opts.append((E.mean(0),E[0],E.max(0),name,r))
for groups in combinations(range(4),1): pass
for m in range(1,16):
 gg=[q for j,q in enumerate([z,b,c,d]) if m>>j&1];
 for scale in [0,1,2]:
  F=np.c_[tuple(gg)] if False else np.column_stack(gg); F=F-F.mean(0);
  if scale==1:F=F/np.maximum(F.std(0),1e-5)
  if scale==2:F=np.arcsinh(F/np.maximum(F.std(0),1e-5));
  A=F@F.T/F.shape[1]; Q=cdist(F,F,'sqeuclidean')/F.shape[1];
  for deg in [1,2,3,4]:test((1+A)**deg,(m,scale,'poly',deg))
  for g in [.03,.12,.4,1.2,3.]:test((1+A)+np.exp(-g*Q),(m,scale,'rbf',g));
  for g in [.2,.65,1.5]:
   r=g*abs(F[:,None,:]-F[None,:,:]); k=(1+r+r*r/3)*np.exp(-r); ak=k.mean(2); pk=ak**2-(k*k).sum(2)/F.shape[1]**2;test((1+A)**2+ak+pk,(m,scale,'add',g))
for col in [0,1]:
 print('best mean',col)
 for e in sorted(opts,key=lambda q:q[0][col])[:12]:print(e)
 print('best gate',col)
 for e in sorted(opts,key=lambda q:q[1][col])[:7]:print(e)
for q in [z,b,c,d,np.c_[z,b,c,d]]:
 a=D(q); pred=a@fit(a,P);print('linear residual / std',np.std(pred-P,0)/np.std(P,0),fit(a,P)[:,0].round(3))
