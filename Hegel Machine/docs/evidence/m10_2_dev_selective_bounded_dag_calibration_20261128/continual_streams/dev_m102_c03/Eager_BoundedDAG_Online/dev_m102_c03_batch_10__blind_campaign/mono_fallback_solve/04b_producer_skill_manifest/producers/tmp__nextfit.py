import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=np.c_[tr['y'],tr['traj']]; t=s['eval_times']; tele=s['telemetry_vec']; np.set_printoptions(precision=6,suppress=True,linewidth=180)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def bias(x):return np.c_[np.ones(len(x)),x]
def amat(z,j):
 p=z[:,j]; q=z[:,(j+1)%4]; return np.c_[p*0+1,p,q,p*q,p**3]
a=s['panel_a']; ix=np.arange(48)
for _ in range(5):
 A=np.array([ls(amat(a['u'][ix],j),a['v'][ix,j]) for j in range(4)]); res=np.linalg.norm(np.column_stack([amat(a['u'],j)@A[j] for j in range(4)])-a['v'],axis=1); ix=np.argsort(res)[:45]
print('A',A,'residual',np.sort(res))
def af(z):return np.column_stack([amat(z,j)@A[j] for j in range(4)])
def ai(z):return least_squares(lambda u:(af(u.reshape(-1,4))-z).ravel(),z.ravel(),max_nfev=25).x.reshape(-1,4)
B=ls(bias(s['panel_b']['u']),s['panel_b']['v']);
def bf(x):return bias(x)@B
def bi(x):return (x-B[0])@np.linalg.inv(B[1:])
from scipy.optimize import minimize_scalar
c=s['panel_c']; u=c['u']; V=c['v1']; z=ai(x)
G={'sin':np.sin,'tanh':np.tanh,'rat':lambda a:a/(1+a*a),'atan':np.arctan,'cube':lambda a:a**3,'lin':lambda a:a,'softroot':lambda a:a/np.sqrt(1+a*a),'erf':__import__('scipy').special.erf}
res=[]
for (n,f),(m,g) in itertools.combinations(G.items(),2):
 for lin in [False,True]:
  def F(U,p):return bias(np.c_[*([U] if lin else []),f(p[0]*U),g(p[1]*U)])
  def fun(p):
   a=F(u,p);return (a@ls(a,V)-V).ravel()
  fit=min((least_squares(fun,p,bounds=(.1,5),max_nfev=80) for p in ([1,1],[1,2.4],[2.4,1],[2.4,2.4])),key=lambda k:k.cost)
  p=fit.x; b=ls(F(u,p),V); res.append((np.std(fit.fun),n,m,lin,p,F(z,p)@b,b))
res.sort(key=lambda a:a[0]); print('C1 winners',[(r[:5]) for r in res[:14]]);print('best matrix',res[0][-1]);
C2=bias(np.cos(2.4*z))@ls(bias(np.cos(2.4*u)),c['v2']);
for e,n,m,lin,p,C1,b in res[:12]:
 W=np.c_[C1,C2]; F=PolynomialFeatures(2).fit_transform(W); Q=F@F.T; R=[]
 for l in [1e-5,.001,.01]:
  E=Q[52:,:52]@np.linalg.solve(Q[:52,:52]+l*np.eye(52),y[:52])-y[52:];R.append(np.r_[np.mean(abs(E[:,0])),np.mean(abs(E[:,1:])),np.max(np.mean(abs(E[:,1:]),0))]);
 print('readout',n,m,lin,p,np.array(R));
# Rank/structure of the target readout, using the best operator fit.
W=np.c_[res[0][5],C2]; wg=PolynomialFeatures(2).fit(W); F=wg.transform(W); names=wg.get_feature_names_out(); _,_,P=np.linalg.svd(y[:52,1:],full_matrices=False); Y=np.c_[y[:,0],y[:,1:]@P[:2].T]; co=ls(F[:52],Y[:52]); print('readout coefficients',list(zip(names,np.round(co,3)))); print('residual sv',np.linalg.svd(Y[:52]-F[:52]@co,compute_uv=False));
