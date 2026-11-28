from probe import *
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from itertools import combinations
z,b,_,w=coords(T['x']); p=S['panel_c']; edges=[(0,1),(0,3),(1,2),(2,3)]
def E(x):return np.column_stack([x[:,i]*x[:,j] for i,j in edges])
c=aff(E(z))@fit(aff(E(p['u'])),p['v1'])
H,A,r=modes(T['traj'][:52]);A=fit(H,T['traj'].T).T;Y=np.c_[T['y'],A]
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
lin=z; quad=power(z,2); edge=E(z); cyc=np.roll(z,-1,axis=1)
F={'z':z,'b':b,'c':c,'w':w,'cosz':np.cos(1.2825*z),'sinz':np.sin(z),'tanhz':np.tanh(z),'ratz':z/(1+z*z),'ratb':b/(1+b*b),'ratw':w/(1+w*w),'sqz':z*z,'edge':edge,'edgecos':np.cos(edge),'edgesin':np.sin(edge),'tanhc':np.tanh(c),'sinc':np.sin(c),'cosc':np.cos(c),'ratc':c/(1+c*c),'pair+':z+cyc,'pair-':z-cyc}
G={}
def add(k,f): G[k]=f
for k,f in F.items():
 add(k,f); add('E*'+k,cross(edge,f)); add('z*'+k,cross(z,f)); add('Cdiag*'+k,c*f)
for k in ('z','b','w','c','pair+','pair-'):
 q=F[k]
 for a in (.1,.2,.5,1.,2.):
  add(f'E/(1+{a}{k}^2)',cross(edge,1/(1+a*q*q)))
  add(f'z/(1+{a}{k}^2)',cross(z,1/(1+a*q*q)))
  add(f'Cdiag/(1+{a}{k}^2)',c/(1+a*q*q))
add('cubic',power(z,3)[:,14:]);add('quartic',power(z,4)[:,34:]);add('Wquad',power(w,2)[:,4:]);add('W*C',cross(w,c));add('W*B',cross(w,b));add('C*B',cross(c,b));add('C*Wdiag',c*w)
for base in [np.c_[quad,w],quad]:
 res=[]
 for k,f in G.items():
  X=aff(np.c_[base,f]); co=fit(X[:52],Y[:52]); yh=X@co; tr=np.mean(abs(yh[:52]-Y[:52]),axis=0);te=np.mean(abs(yh[52:]-Y[52:]),axis=0)
  h=np.sum(X[:52]*np.linalg.pinv(X[:52]).T,axis=1);loo=np.mean(abs((yh[:52]-Y[:52])/np.maximum(1-h,1e-8)[:,None]),axis=0)
  res.append((k,tr,te,loo))
 print('base size',base.shape[1]);
 for j in range(3):
  for k,tr,te,loo in sorted(res,key=lambda v:v[3][j])[:10]: print(j,k,np.round([tr[j],loo[j],te[j]],5))
# Check the shared nonlinear component by removing the quadratic part on training rows.
for base in (quad,np.c_[quad,w],np.c_[quad,w,z*w]):
 X=aff(base); ee=Y-X@fit(X[:52],Y[:52]);print('residual std',ee[:52].std(0), 'residual correlation',np.corrcoef(ee[:52].T).round(4))
 for a,b in [(0,2),(1,2),(0,1)]:
  X=aff(np.c_[base,Y[:,b]]); e=X@fit(X[:52],Y[:52,a])-Y[:,a]; print('oracle',a,b,np.mean(abs(e[:52])),np.mean(abs(e[52:])))
