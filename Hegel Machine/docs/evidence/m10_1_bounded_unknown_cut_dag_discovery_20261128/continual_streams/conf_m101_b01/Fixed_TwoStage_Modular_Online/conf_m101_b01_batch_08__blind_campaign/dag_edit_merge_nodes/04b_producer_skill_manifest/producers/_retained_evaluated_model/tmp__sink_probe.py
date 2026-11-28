from merged_probe import *
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
x=S['train']['x']; y=S['train']['y']; t=S['train']['traj']; Y=np.c_[y,t]; u,_=a_inv(x,a_fit(S['panel_a'])); b=b_fit(S['panel_b']); c,d=c_fit(S['panel_c']); P,Q,R=b(u),c(u),d(u)
for i in [0,10,20,30,40,50,60,70]: print('row',i,np.round(u[i],3),np.round(P[i],3),np.round(Q[i],3),np.round(R[i],3),np.round(Y[i,[0,1,3,8]],3))
N=len(u); splits=[(np.arange(54),np.arange(54,72))]+list(KFold(4,shuffle=True,random_state=17).split(u))
def cv(K):
 E=np.array([np.mean(abs(K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)],Y[i])-Y[j]),axis=0) for i,j in splits]); return np.round([E[0,0],np.mean(E[0,1:]),max(E[:,0]),max(np.mean(E[:,1:],axis=1))],4)
for label,F in [('raw',x),('u',u),('u+B',np.c_[u,P]),('u+C',np.c_[u,Q,R]),('all',np.c_[u,P,Q,R]),('composed',np.c_[u,P,Q,R,b(R),c(P),d(P),c(R),d(Q),b(Q)])]:
 F=(F-F.mean(0))/F.std(0)/np.sqrt(F.shape[1]); dist=cdist(F,F,'sqeuclidean'); G=F@F.T; I=np.eye(N); print(label)
 for k in [1,2,3,4,5,6]: print('poly',k,cv((1+G)**k+1e-4*I))
 for l in [.5,1,2,4]:
  r=np.sqrt(5*dist)/l; print('mat',l,cv((1+r+r**2/3)*np.exp(-r)+1e-4*I),'rbf',cv(np.exp(-dist/(2*l*l))+1e-4*I))
# A single exponential response, optionally with an initial amplitude, tested at the sample level.
tm=S['eval_times']; tr=t
for form in [0,1,2]:
 E=[]; params=[]
 for row in tr:
  def design(k):
   q=np.exp(-k*tm); return np.c_[1-q,q] if form==0 else ((1-q)[:,None] if form==1 else np.c_[1-q,tm])
  def loss(k):
   F=design(k); return np.sum((F@ls(F,row)-row)**2)
  k=minimize_scalar(loss,bounds=(.05,3),method='bounded').x; pars=ls(design(k),row); E.append(loss(k)); params.append(np.r_[pars,k])
 pars=np.array(params); print('trajectory form',form,'rmse',np.sqrt(np.mean(E)/len(tm)),'quantiles',np.quantile(pars,[0,.1,.5,.9,1],axis=0))
# Test whether B is an odd saturation plus a simple even correction.
p=S['panel_b']; z=p['u']; v=p['v']; polys={'exp':lambda u,k:u*np.exp(-k*u*u),'erf':lambda u,k:__import__('scipy').special.erf(k*u),'tanh':lambda u,k:np.tanh(k*u),'rat':lambda u,k:u/(1+k*u*u),'sin':lambda u,k:np.sin(k*u)}
for label,f in polys.items():
 for even in [False,True]:
  def res(p):
   g=z+p[0]*f(z,p[1]); F=np.c_[ones(z),g,z*z] if even else np.c_[ones(z),g]; return (F@ls(F,v)-v).ravel()
  vals=[least_squares(res,[-.7,k],bounds=([-10,.1],[10,8])) for k in [.6,1.5,3.]]; opt=min(vals,key=lambda r:np.linalg.norm(r.fun)); print('B even scan',label,even,np.std(opt.fun),opt.x)
