from probe import *
from itertools import combinations_with_replacement

def ls(a,y):return np.linalg.lstsq(a,y,rcond=1e-9)[0]
def ones(x):return np.c_[np.ones(len(x)),x]
def pair(x):return x*np.roll(x,-1,axis=1)
w1=ls(ones(pair(u)),v1)
w2=ls(ones(np.c_[u,u*u]),v2)
c=lambda x:ones(pair(x))@w1
d=lambda x:ones(np.c_[x,x*x])@w2
H=lambda k:np.c_[np.ones(len(t)),-np.expm1(-k[0]*t),-np.expm1(-k[1]*t)]
k=least_squares(lambda k:(H(k)@ls(H(k),tr.T)-tr.T).ravel(),[.4,1.6],bounds=([.05,.6],[1.,5.])).x
Y=np.c_[y,ls(H(k),tr.T).T]; print('STUDY',s.keys(),'TIMES',t,'rates',k,'corr',np.corrcoef(Y.T),'std',Y.std(0))
a=np.arange(52); j=np.arange(52,72)
def report(tag,F):
 F=ones(F); w=ls(F[a],Y[a]); r=F@w-Y
 print(tag,F.shape[1],np.round(np.mean(abs(r[a]),0),4),np.round(np.mean(abs(r[j]),0),4),'traj',round(np.mean(abs(r[j,1:]@H(k).T)),5))
for tag,x in [('b',b),('z',z),('zb',np.c_[z,b]),('zbc(b)',np.c_[z,b,c(b)]),('zbc(z)',np.c_[z,b,c(z)]),('bcd(b)',np.c_[b,c(b),d(b)]),('zbc(b)d(z)',np.c_[z,b,c(b),d(z)])]:
 report(tag,x)
 if tag in ('b','z','zb'):
  m=np.column_stack([x[:,i]*x[:,l] for i,l in combinations_with_replacement(range(x.shape[1]),2)])
  report(tag+' quad',np.c_[x,m]); print('coef linear',np.round(ls(ones(x),Y),3))
# Test saturating a projected coordinate, rather than saturating each input separately.
for tag,x in [('b',b),('z',z),('bc',np.c_[b,c(b)]),('zbc',np.c_[z,b,c(b)]),('zbcd',np.c_[z,b,c(b),d(z)])]:
 F=ones(x); P=np.linalg.pinv(F[a]); best=[]
 for l in range(4):
  yy=Y[a,l]; init=P@yy; scores=[]
  for fn,ini in [(np.tanh,init),(np.sin,init),(lambda q:q*q,init),(lambda q:q/(1+q*q),init)]:
   def m(w,i):return np.c_[F[i],fn(F[i]@w)]
   def res(w):
    M=m(w,a);return M@ls(M,yy)-yy
   w=least_squares(res,ini,max_nfev=180).x; out=m(w,j)@ls(m(w,a),yy);scores.append((round(np.mean(abs(res(w))),4),round(np.mean(abs(out-Y[j,l])),4)))
  best.append(scores)
 print('singleindex',tag,best)
