exec(open('/tmp/read14.py').read().split('for f in (1.25')[0])
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
U=b['u']; V=b['v']; y=s['train']['y']; T=s['train']['traj']; Y=np.c_[y,T]; n=52

def probe(U,V,fn, name):
 def D(x,f):return aff(np.c_[x,fn(x*f)])
 def err(f):
  A=D(U,f);return (A@fit(A,V)-V).ravel()
 opt=least_squares(err,np.full(4,2.),bounds=(.15,5.));print(name,opt.x,np.std(opt.fun));print(np.round(fit(D(U,opt.x),V),4));return lambda x:D(x,opt.x)@fit(D(U,opt.x),V)
B1=probe(U,V,np.sin,'Bsin');B2=probe(U,V,np.tanh,'Btanh');B3=probe(U,V,lambda x:x/(1+x*x),'Brat');C=s['panel_c'];C1=probe(C['u'],C['v1'],np.tanh,'C1');C2=probe(C['u'],C['v2'],np.cos,'C2');

def poly(x,d):return np.column_stack([np.prod(x[:,q],axis=1) for q in cr(range(x.shape[1]),d)])
def met(D,reg=0):
 A=D[:n]; Y0=Y[:n];
 if reg:
  m=A.mean(0); scale=np.maximum(A.std(0),1e-8); A=(A-m)/scale; DD=(D[n:]-m)/scale; p=DD@np.linalg.solve(A.T@A+reg*np.eye(A.shape[1]),A.T@(Y0-Y0.mean(0)))+Y0.mean(0)
 else:p=D[n:]@fit(A,Y0)
 E=np.abs(Y[n:]-p);return np.round([E[:,0].mean(),E[:,1:].mean()],5)
for tag,z in [('sin',B1(u)),('tanh',B2(u)),('rat',B3(u)),('u',u)]:
 for tag2,v in [('z',z),('C1z',C1(z)),('C1u',C1(u)),('C2z',C2(z)),('C2u',C2(u))]:
  A=aff(v); Q=np.c_[A,poly(v,2)]; P=np.c_[Q,poly(v,3)];print(tag,tag2,met(A),met(Q),met(P),met(P,.4))
 print('joint',tag,met(np.c_[aff(z),C1(z),C2(z)]),met(np.c_[aff(z),C1(u),C2(u)]))
