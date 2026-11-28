from probe import *
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
I=np.arange(len(x)); Y=np.c_[y,z]; main=[I[56:],*np.array_split(np.random.default_rng(18).permutation(I),4),np.argsort(np.linalg.norm(x,axis=1))[-16:]]; edges=[np.argsort(sign*x[:,j])[-16:] for j in range(4) for sign in (1,-1)]
R=aff(aff(x)@A)@B
choices=[('baseline',0,0,0),*[('rbf',a,b,0) for a in (.45,1.0,1.8,3.2) for b in (.035,.3,2.)],*[('p4',a,b,0) for a in (.05,.3,1.,3.) for b in (0.,.2,1.)],*[('mix',a,b,c) for a in (.2,.8,2.) for b in (.05,.35,2.) for c in (.65,1.4,2.7)]]
def kernel(it):
 W=R/R[it].std(0); G=(W@W.T)/4; D=cdist(W,W,'sqeuclidean')/4; f=poly(R,2); Q=f@f.T; out=[]
 for k,a,b,c in choices:
  if k=='baseline': M=Q+.3*np.exp(-1.3*D)
  if k=='rbf': M=Q+b*np.exp(-a*D)
  if k=='p4': M=Q+a*G**3+a*b*G**4
  if k=='mix': M=Q+a*G**3+b*np.exp(-c*D)
  out.append(M)
 return out

def temporal(it):
 def b(p):return np.c_[np.exp(-p[0]*t)*np.sin(p[1]*t),-np.expm1(-p[2]*t)]
 def er(p):
  f=b(p);return (f@fit(f,z[it].T)-z[it].T).ravel()
 f=b(least_squares(er,[.2,1.1,.55],bounds=([0,.5,.08],[1,2,2])).x);return f@np.linalg.pinv(f)

def run(it,ho):
 target=Y[it].copy(); target[:,1:]=target[:,1:]@temporal(it); out=[]
 for M in kernel(it):
  P=M[np.ix_(ho,it)]@cho_solve(cho_factor(M[np.ix_(it,it)]+1e-5*np.eye(len(it))),target); e=abs(P-Y[ho]);out.append(np.c_[e[:,0],e[:,1:].mean(1),e[:,-1]].mean(0))
 return np.array(out)
all_e=[]; nested=[]
for h in main+edges:
 it=np.setdiff1d(I,h); e=run(it,h); all_e.append(e)
 folds=np.array_split(np.random.default_rng(721).permutation(it),4)+[it[np.argsort(np.linalg.norm(x[it],axis=1))[-13:]]]
 inn=np.array([run(np.setdiff1d(it,h2),h2) for h2 in folds]); sc=inn.mean(0)+.4*inn.max(0); sy=sc[:,0]; st=sc[:,1]+.45*sc[:,2]; ii=np.argsort(sy)[:4]; jj=np.argsort(st)[:4]; nested.append(e[np.array([ii[0],jj[0],jj[0]]),np.arange(3)]); print('split',len(all_e), 'base',e[0].round(5),'nested',nested[-1].round(5),'y',choices[ii[0]],'traj',choices[jj[0]])
e=np.array(all_e); nested=np.array(nested);print('nested main mean/worst and edge worst',nested[:6].mean(0), nested[:6].max(0),nested[6:].max(0));
for j in range(3):
 score=e[:6,:,j].mean(0)+.35*e[:6,:,j].max(0)+.25*e[6:,:,j].max(0)
 for i in np.r_[0,np.argsort(score)[:8]]:print(j,choices[i],np.r_[e[0,i,j],e[:6,i,j].mean(),e[:6,i,j].max(),e[6:,i,j].max()].round(5))
