from operators import *
from itertools import product
b=ops['B'](z); c=ops['C'](z); d=ops['D'](z)
a,f,g,h=b.T; pair=a*g+f*h; contrast=a+f-g-h
print('ARCH GRAMMAR',scores(np.c_[aff(b),pair,np.tanh(contrast),pair/(1+contrast**2)])); print('B affine',scores(aff(b)))
for name,x in [('z',z),('B',b),('C',c),('D',d)]:
 print('POLY COEFF',name,np.c_[fit(pol(x,2)[:72],Y[:,0]),fit(pol(x,2)[:72],Y[:,-1])].round(4).T)
F=[]; names=[]
def add(n,a):
 names.append(n);F.append(a)
for k,x in [('B',b),('z',z),('C',c),('D',d)]:
 Q=pol(x,3)
 for j in range(1,len(mon)):add((k,mon[j]),Q[:,j])
 for q in product((0,1,-1),repeat=4):
  if not any(q) or next(v for v in q if v)!=1:continue
  u=x@q
  for rate in (.4,.7,1,1.4,1.7,2.4,3.4):
   for tag,v in [('tanh',np.tanh(rate*u)),('sin',np.sin(rate*u)),('cos',np.cos(rate*u)-1)]:add((k,q,rate,tag),v)
   for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(-1,-1)):
    v=x[:,i]*x[:,j] if i>=0 else x[:,0]*x[:,2]+x[:,1]*x[:,3];add((k,q,rate,'rat',i,j),v/(1+rate*u*u))
F=np.array(F).T
for rs in [range(52),range(20,72)]:
 ix=np.array(rs); jj=np.setdiff1d(np.arange(72),ix)
 for target in (0,1,-1):
  print('GREEDY split/target',ix[0],target); y=Y[:,target]; base=aff(b); picks=[]
  for i in range(10):
   Q=np.linalg.qr(base[ix])[0]; rem=F[ix]-Q@(Q.T@F[ix]); r=y[ix]-Q@(Q.T@y[ix]); j=np.argmax(abs(r@rem)/np.maximum(np.linalg.norm(rem,axis=0),1e-10)); picks.append(j); base=np.c_[aff(b),F[:,picks]]; co=fit(base[ix],y[ix]); print(names[j],round(co[-1],4),round(np.mean(abs(base[jj]@co-y[jj])),5))
