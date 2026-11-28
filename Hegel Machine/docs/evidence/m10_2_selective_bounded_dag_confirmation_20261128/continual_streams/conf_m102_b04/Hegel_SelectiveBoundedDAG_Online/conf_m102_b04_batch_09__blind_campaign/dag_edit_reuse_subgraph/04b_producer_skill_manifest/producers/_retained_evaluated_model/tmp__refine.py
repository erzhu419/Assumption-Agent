from screen import *
from scipy.optimize import least_squares
from itertools import combinations
c=s['panel_c']; u=c['u']; V=c['v1'];
for nm,fn in [('atan',np.arctan),('asinh',np.arcsinh),('tanh',np.tanh)]:
 for ex in ['none','lin','cube','square','cyclic','linear-cyclic']:
  def basis(q,p):
   z=[aff(fn(p*q))];
   if ex in ('lin','linear-cyclic'):z.append(q)
   if ex=='cube':z.append(q**3)
   if ex=='square':z.append(q**2)
   if ex in ('cyclic','linear-cyclic'):z.append(q*np.roll(q,-1,axis=1))
   return np.column_stack(z)
  def err(p):
   D=basis(u,p);return np.mean((D@np.linalg.lstsq(D,V,rcond=None)[0]-V)**2)
  o=minimize_scalar(err,bounds=(.1,6),method='bounded');print(nm,ex,o.x,o.fun**.5)
D=-np.expm1(-s['eval_times'][:,None]*np.array([.42,1.8])); amps=np.linalg.lstsq(D,t.T,rcond=None)[0].T
Q=np.c_[coords['x'],coords['b'],coords['v'],coords['w'],coords['vb'],coords['wb']]; names=[k+str(i) for k in ('x','b','v','w','vb','wb') for i in range(4)]; A=np.c_[np.ones(n),np.arctan(2.1*x),x,x*x]; R=lambda v:v-A@np.linalg.lstsq(A,v,rcond=None)[0]; Z=np.c_[y,amps]; print('base residuals',np.std(R(Z),0),'raw std',Z.std(0));
fs=[]; nn=[]
for i,z in enumerate(Q.T):
 for nm,fn in [('p2',lambda z:z*z),('p3',lambda z:z**3),('sin',np.sin),('exp',lambda z:np.expm1(-z*z)),('tanh',np.tanh),('atan',np.arctan)]:
  for m in [.5,1.,2.,3.]:fs.append(fn(m*z));nn.append((nm,m,names[i]))
for i,j in combinations(range(Q.shape[1]),2):
 a,b=Q[:,i],Q[:,j]
 for nm,f in [('p',a*b),('rat',a*b/(1+a*a+b*b)),('a2b',a*a*b),('ab2',a*b*b)]:fs.append(f);nn.append((nm,names[i],names[j]))
F=np.column_stack(fs); rf=R(F); rz=R(Z)
for j in range(3):
 sc=abs(rf.T@rz[:,j])/np.maximum(np.linalg.norm(rf,axis=0),.01)/np.linalg.norm(rz[:,j]);print('partial',j,[(nn[k],round(sc[k],3)) for k in np.argsort(sc)[-12:][::-1]])
print('assets');print(open('/holdout_task/public_asset_index.json').read()[:2000]);print('archive',open('/tmp/hegel_continual_archive/archive_manifest.json').read()[:1300]);print(s.keys())
