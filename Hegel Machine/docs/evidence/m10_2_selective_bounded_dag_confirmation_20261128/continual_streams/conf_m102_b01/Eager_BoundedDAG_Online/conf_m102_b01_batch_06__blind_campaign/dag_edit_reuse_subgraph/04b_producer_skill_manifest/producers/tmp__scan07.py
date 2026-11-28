exec(open('/tmp/probe07.py').read().split('for name,x')[0])
from itertools import product
from scipy.optimize import least_squares
for key,dct in [('b',b),('c',c)]:
 for k in dct.keys()-{'u','v1'}:
  x=dct['u']; z0=dct[k]; M=poly(x,2); C=fit(M,z0); print(key,k,'quadratic rms',np.std(M@C-z0)); print('squares dependence',np.round([C[5+i] if False else C[5+sum(4-h for h in range(i))]/C[1+i] for i in range(4)],3)); print('cubic rms',np.std(poly(x,3)@fit(poly(x,3),z0)-z0));
  for name,F in [('quadsep',lambda x:np.c_[aff(x),x*x]),('cubsep',lambda x:np.c_[aff(x),x*x,x**3]),('sinmix',lambda x:np.c_[aff(x),np.sin(x),np.cos(x)]),('tanhmix',lambda x:np.c_[np.ones(len(x)),np.tanh(x),np.tanh(x)**2]),('expmix',lambda x:np.c_[aff(x),np.exp(x)]),('logs',lambda x:np.c_[aff(x),np.logaddexp(0,x)]),('rational',lambda x:np.c_[aff(x),x*x/(1+x*x)])]:
   H=F(x); h=H@np.linalg.pinv(H); e=z0-h@z0; print(name,np.std(e),np.mean(abs(e/(1-np.diag(h)[:,None]))));
print('b affine residual rows',np.round(np.linalg.norm(aff(b['u'])@fit(aff(b['u']),b['v'])-b['v'],axis=1),4)); print('sv',np.linalg.svd(T,compute_uv=False)); print('Y vs T residual',np.std(y-aff(T)@fit(aff(T),y)))
keep=np.ones(len(b['u']),bool)
for _ in range(10):
 D=aff(b['u']); Cb=fit(D[keep],b['v'][keep]); e=np.linalg.norm(D@Cb-b['v'],axis=1); keep=e<max(0.001,3*np.median(e));
print('robust B',keep.sum(),np.round(e,4)); wb=aff(u)@Cb
xall={'x':raw,'u':u,'w':w,'rb':wb,'b2':B(u)}
for name,x in list(xall.items()):
 for actname,act in [('tanh',np.tanh),('atan',np.arctan),('asinh',np.arcsinh),('inv3',lambda x:np.sinh(np.arcsinh(x)/3)),('sin',np.sin)]:
  for g in (1.,2.,4.,8.):xall[f'{actname}{g:g}({name})']=act(g*x)
rows=[]
for name,x in xall.items():
 for label,base in [('c1',P(x)),('c2',Q(x)),('c1c2',np.c_[P(x),Q(x)]),('p2',poly(x,2)[:,1:])]:
  for nameg,g in [('id',lambda x:x),('sat',np.tanh),('atan',np.arctan),('asinh',np.arcsinh)]:
   for base2 in (False,True):
    X=aff(np.c_[g(base),x] if base2 else g(base)); a0,a1=score(X); rows.append((*a0,*a1,name,label,nameg,base2));
for key in (0,1,2,3):
 print('BEST',key); print(*sorted(rows,key=lambda q:q[key])[:9],sep='\n')
