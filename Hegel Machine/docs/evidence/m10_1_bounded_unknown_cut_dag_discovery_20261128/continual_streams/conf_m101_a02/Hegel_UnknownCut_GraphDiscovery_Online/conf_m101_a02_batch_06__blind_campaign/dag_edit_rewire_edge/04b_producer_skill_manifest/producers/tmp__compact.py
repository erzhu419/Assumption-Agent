from probe import *
from sklearn.model_selection import KFold
from scipy.linalg import solve
V=x*np.roll(z,1,1); q=z@[-1,-1,1,1]; w=V.sum(1); base=np.c_[np.ones(len(x)),q,w]; F0=np.c_[np.ones(len(x)),z,V]; R=np.c_[y,T]; es=[]
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=281).split(x)); a,b=folds[0]
def test(F):
 return np.array([np.abs(F[i]@np.linalg.lstsq(F[j],R[j],rcond=None)[0]-R[i]).mean(0) for j,i in folds])
def report(F):
 e=test(F);return np.r_[e[0,0],e[1:,0].mean(),e[1:,0].max(),e[0,1:].mean()]
# Temporal modes: distinguish common exponential-plus-forcing from horizon-specific noise.
for B1 in [lambda k:np.c_[-np.expm1(-k*t),t],lambda k:np.c_[-np.expm1(-k*t),np.log1p(t)],lambda k:np.c_[-np.expm1(-k[0]*t),-np.expm1(-k[1]*t)],lambda k:np.c_[-np.expm1(-k*t),t,np.ones(len(t))]]:
 def calc(p):
  f=B1(p if len(p)==2 else p[0]);return T@np.linalg.pinv(f.T)@f.T-T
 try:
  r=least_squares(lambda p:calc(p).ravel(),[.6],bounds=(.01,6));print('timefit',r.x,np.mean(abs(r.fun)))
 except IndexError:
  r=least_squares(lambda p:calc(p).ravel(),[.6,1.3],bounds=(.01,6));print('timefit',r.x,np.mean(abs(r.fun)))
# Scan bounded univariate corrections to the observed cyclic/contrast law.
sc={}
for na,M in [('x',x),('z',z),('v',V),('diag',x*z),('next',x*np.roll(z,-1,1)),('opp',x*np.roll(z,2,1)),('xx',x*np.roll(x,1,1)),('zz',z*np.roll(z,1,1))]:
 for k in range(4):sc[na+str(k)]=M[:,k]
 for sg in itertools.product([-1,0,1],repeat=4):
  non=np.flatnonzero(sg)
  if len(non)>=2 and sg[non[0]]==1:sc[na+str(sg)]=M@sg
out=[]
for bn,B in [('tied',base),('free',F0)]:
 for name,v in sc.items():
  for form,h in [('lin',v),('square',v*v),('sin',np.sin(v)),('cos',np.cos(v)),('sat',np.tanh(v)),('wr',w/(1+v*v)),('qr',q/(1+v*v)),('qv',q*v),('wv',w*v)]:
   F=np.c_[B,h]; e=report(F);out.append((e,bn,name,form))
print('corrections')
for item in sorted(out,key=lambda it:it[0][1]+.9*it[0][2]+.3*it[0][0])[:25]:print(item)
for bn,B in [('tied',base),('free',F0)]:
 for pairs in [np.roll(z,2,1)[:,:2]*z[:,:2],z*z,np.cos(1.27*x),np.c_[x*x,z*z],np.c_[x*np.roll(z,-1,1),x*z],z/(1+z*z),x/(1+x*x)]:
  print('family',bn,report(np.c_[B,pairs]))
# Residual correlations reveal whether the omitted signal tracks trajectory amplitudes or input extremes.
coef=np.linalg.lstsq(F0[a],R[a],rcond=None)[0]; residual=R-F0@coef
print('resid',np.c_[q,w,residual[:,0],residual[:,-1]][np.argsort(residual[:,0])[[0,1,2,-3,-2,-1]]].round(4)); print('linear',coef[:,[0,-1]].round(4));print('joint',np.corrcoef(residual.T)[0].round(3))
