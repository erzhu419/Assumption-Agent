import numpy as np,pickle
from itertools import combinations,product
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; F=np.load('/tmp/coords.npz'); z,b,c,d,V=[F[k] for k in ('z','b','c','d','V')]; n=52
ls=lambda X,Y:np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(q):return PolynomialFeatures(2).fit_transform(q)
def metric(E):return np.mean(abs(E[:,0])),np.mean(abs(E[:,1:]))
fun={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'r':lambda x:x/(1+x*x),'h':lambda x:x/np.sqrt(1+x*x)}
coords={'z':z,'b':b,'c':c,'d':d}; rows=[]
def evaluate(A,name):
 B=ls(A[:n],V[:n]); E=A@B-V; rows.append((*metric(E[n:]),*metric(E[:n]),name))
for nam,q in coords.items():
 for gn,g in fun.items():
  for k in [.22,1,1.25,1.8,2.15,2.4,2.85,3.5,4.3]:
   u=g(k*q); evaluate(np.c_[poly(q),u],(nam,gn,k,'add')); evaluate(poly(u),(nam,gn,k,'transform')); evaluate(np.c_[poly(q),u,u*q],(nam,gn,k,'local'))
 for nam2,q2 in coords.items():
  if nam2!=nam:
   for typ,D in [('lin',q2),('quad',poly(q2)[:,1:]),('prod',(q[:,:,None]*q2[:,None,:]).reshape(len(q),16))]:evaluate(np.c_[poly(q),D],(nam,nam2,typ))
print('fixed y',sorted(rows)[:10]); print('fixed t',sorted(rows,key=lambda a:a[1])[:10],flush=True)
rows=[]
W=V[:,[0,-1]]
for name,q in coords.items():
 for fn,g in fun.items():
  for mode in range(5):
   P=poly(q)
   def feat(a):
    h=g(a[0]*q+a[1]);
    if mode==0:return np.c_[P,h]
    if mode==1:return poly(q+a[2]*h)
    if mode==2:return poly(h)
    if mode==3:return np.c_[P,h,h*q]
    return np.c_[aff(q),h,poly(q+a[2]*h)[:,5:]]
   def f(a):
    A=feat(a)[:n];return (A@ls(A,W[:n])-W[:n]).ravel()
   for k in [1.1,2.2,3.5]:
    for h in ([.4,-.4] if mode in (1,4) else [0]):
     o=least_squares(f,[k,0,h],max_nfev=90,bounds=([.12,-3.2,-5],[7,3.2,5]));evaluate(feat(o.x),(name,fn,mode,o.x))
print('profiled y',sorted(rows)[:14]);print('profiled t',sorted(rows,key=lambda a:a[1])[:14]);print('profiled train',sorted(rows,key=lambda a:a[2])[:10])
