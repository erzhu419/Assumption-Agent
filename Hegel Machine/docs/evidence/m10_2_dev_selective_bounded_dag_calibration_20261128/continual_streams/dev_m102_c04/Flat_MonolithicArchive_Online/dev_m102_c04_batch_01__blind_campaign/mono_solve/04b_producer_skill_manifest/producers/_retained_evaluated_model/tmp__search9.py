from dag import *
from itertools import combinations
from scipy.linalg import qr
v=np.c_[y,t]; print('y from t',np.mean(abs(poly(t[52:],1)@ls(poly(t[:52],1),y[:52])-y[52:])),flush=True)
print('ca',ca,'summary check',y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std(),flush=True)
k,c=s['telemetry_vec'][:2]; N={'T':lambda z:np.tanh(k*z),'S':lambda z:np.sin(k*z),'C':lambda z:np.cos(k*z),'R':lambda z:z/(1+c*z*z),'D':lambda z:z/(1+c*abs(z)),'E':lambda z:z*z/(1+c*z*z),'U':lambda z:z*np.sqrt(1+c*z*z),'J':lambda z:z/(1+k*abs(z)),'K':lambda z:np.tanh(k*z)/(1+c*z*z),'P':lambda z:z+c*z**3,'F':lambda z:np.sin(k*z+c*z*z),'W':lambda z:z/(1+c*np.sum(z*z,1,keepdims=True))}; O={'B':B,'b':Bi,'Q':Q,'H':H}; out=[]
def test(name,f):
 f=np.c_[np.ones(len(x)),f]; a=f[:52]; w=ls(a,v[:52]); e=np.abs(f@w-v);out.append((np.mean(e[:52,0]),np.mean(e[52:,0]),np.mean(e[:52,1:]),np.mean(e[52:,1:]),name))
for n,z in paths(x,2).items():
 for j,fn in N.items():
  w=fn(z)
  test(n+j+'lin',np.c_[z,w]);test(n+j+'quad',np.c_[z,poly(w,2)[:,1:]]);test(n+j+'cub',np.c_[z,poly(w,3)[:,1:]])
  for a,op in O.items():
   h=op(w)
   test(n+j+a,np.c_[x,z,w,h]);test(n+j+a+'prod',np.c_[x,z,w,h,w*h,z*h]);test(n+j+a+'quad',np.c_[z,w,poly(h,2)[:,1:]])
   if len(n)<=2:
    for b,op2 in N.items():
     hh=op2(h);test(n+j+a+b,np.c_[x,z,w,h,hh]);test(n+j+a+b+'prod',np.c_[x,z,w,h,hh,w*hh,h*hh]);
for n,z in paths(x,1).items():
 Qz,HZ=Q(z),H(z)
 for j,w in [('z',z),*((j,fn(z)) for j,fn in N.items())]:
  for a,h in [('q',Q(w)),('h',H(w)),('mix',Q(w)+H(w)),('b',B(w))]:
   for b,l in [('q',Q(w)),('h',H(w)),('z',z),('B',B(w)),('w',w),('1',np.ones_like(w))]:
    for label,m in [('rat',h/(1+c*l*l)),('rabs',h/(1+c*abs(l))),('sat',np.tanh(k*h)/(1+c*l*l)),('mix',np.tanh(k*(h+c*l))),('f',np.sin(k*(h+c*l))),('dotrat',h/(1+c*np.sum(l*l,1,keepdims=True)))]:
     test(n+j+a+b+label,np.c_[z,w,h,l,m]);test(n+j+a+b+label+'prod',np.c_[z,w,h,l,m,h*l,w*m]);
for i in range(4):print(i,*sorted(out,key=lambda a:a[i])[:8],sep='\n')
