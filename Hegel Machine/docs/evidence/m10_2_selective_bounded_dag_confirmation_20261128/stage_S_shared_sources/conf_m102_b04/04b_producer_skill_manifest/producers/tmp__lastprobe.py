import numpy as np,pickle
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; v=p['v1']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; m=np.c_[y,z]; I=np.arange(len(x)); H=lambda x:np.c_[np.ones(len(x)),x]
def soft(q): return q/np.sqrt(1+q*q)
models=[]
for name,g in [('tanh',np.tanh),('atan',np.arctan),('soft',soft),('sin',np.sin),('erf',__import__('scipy.special',fromlist=['erf']).erf),('asinh',np.arcsinh),('rational',lambda q:q/(1+q*q)),('log',lambda q:np.sign(q)*np.log1p(abs(q)))]:
 for typ in ['independent','shared']:
  def feat(x,q):
   f=g(x*q[0]);return np.c_[H(x),f] if typ=='independent' else H(x+q[1]*f)
  def fun(q):
   F=feat(u,q); w=np.linalg.lstsq(F,v,rcond=None)[0];return (F@w-v).ravel()
  sol=min((least_squares(fun,[k,1] if typ=='shared' else [k],bounds=([.03,-20],[8,30]) if typ=='shared' else ([.03],[8])) for k in [.5,1.5,2.5,4]),key=lambda a:a.cost)
  rm=np.mean(sol.fun**2)**.5;print('map',name,typ,sol.x,rm);models.append((rm,name,typ,sol.x.copy(),g))
a=s['panel_a']; f=H(a['u']); W=np.linalg.lstsq(f,a['v'],rcond=None)[0]
for _ in range(10):
 j=np.argsort(np.linalg.norm(f@W-a['v'],axis=1))[:39];W=np.linalg.lstsq(f[j],a['v'][j],rcond=None)[0]
c=s['panel_c']; b=s['panel_b']; wb=np.linalg.lstsq(H(b['u']),b['v'],rcond=None)[0]; Fa=lambda h:np.c_[H(h),np.logaddexp(1.1*h,-1.1*h)-np.log(2)]; w2=np.linalg.lstsq(Fa(u),c['v2'],rcond=None)[0];
folds=[I[52:],np.argsort(np.linalg.norm(x,axis=1))[-20:],np.argsort(abs(x).max(1))[-20:],I[:20]]+[np.random.default_rng(i).choice(len(x),20,False) for i in range(8)]+[np.argsort(sign*x[:,j])[-20:] for j in range(4) for sign in [1,-1]]
P=lambda a,d:PolynomialFeatures(d).fit_transform(a); rows=[]
def add(label,F):
 K=F@F.T
 for a in [1e-7,.0001,.001,.007,.02,.06]:
  E=[]
  for te in folds:
   tr=np.setdiff1d(I,te); d=abs(K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+a*np.eye(len(tr)),m[tr])-m[te]);E.append([d[:,0].mean(),d[:,1:].mean(0).max()])
  E=np.array(E);rows.append((label,a,E[0],E.max(0),E.mean(0)))
for _,name,typ,q,g in sorted(models,key=lambda r:r[0])[:3]:
 def feat(h):
  f=g(h*q[0]);return np.c_[H(h),f] if typ=='independent' else H(h+q[1]*f)
 w=np.linalg.lstsq(feat(u),v,rcond=None)[0]
 for nm,X in [('raw',x),('inv',(x-W[0])/W[1:].diagonal()),('dir',x*W[1:].diagonal()+W[0])]:
  C=feat(X)@w; V=Fa(X)@w2; B=H(X)@wb; PC=np.linalg.lstsq(H(u),v,rcond=None)[0]; A=soft(.3*(H(X)@PC))/.3
  base=f'{name}/{typ}/{nm}'
  for lab,F in [('raw',P(X,3)),('proxy',P(A,3)),('C2',P(C,2)),('C3',P(C,3)),('C+V',np.c_[P(C,2),V]),('CV2',P(np.c_[C,V],2)),('XV2',P(np.c_[X,V],2)),('XC2',P(np.c_[X,C],2)),('B+CV',np.c_[P(B,2),P(np.c_[C,V],2)]),('XC+V',np.c_[P(np.c_[X,C],2),V]),('raw+C',np.c_[P(X,3),P(C,2)]),('proxy+C',np.c_[P(A,3),P(C,2)]),('proxy+raw',np.c_[P(A,3),P(X,3)])]:add(base+'/'+lab,F)
for j in range(2):
 print('TARGET',j)
 for name,r,p,m,a in sorted(rows,key=lambda e:.5*e[3][j]+e[4][j])[:12]: print(name,r,np.round(p,5),np.round(m,5),np.round(a,5))
