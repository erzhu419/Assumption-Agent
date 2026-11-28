import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m101_b04_batch_04_solver as old
from scipy.linalg import qr
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=S['train']; n=len(tr['y']); za=old._calibrate(S['panel_a'],np.r_[tr['x'],S['test']['x']]);
def fit(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0]
def poly(x,d):
 idx=[i for j in range(d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),j)]
 return np.column_stack([np.prod(x[:,i],axis=1) for i in idx]),idx
def cvfit(X,Y):
 r=Y-X@fit(X,Y); q=qr(X,mode='economic')[0];return np.sqrt(np.mean((r/(1-(q*q).sum(1))[:,None])**2))
def pa(t):
 return np.column_stack((np.ones(len(t)),t,np.sin(t),t**3))
ops={}
if __name__=='__main__':
 for k in ['panel_b','panel_c']:
  d=S[k];u=d['u']; A=poly(u,1)[0];Q=poly(u,2)[0];C=np.c_[Q,u**3];H=np.c_[A,np.sin(1.7*u),np.cos(1.7*u)]
  for name,v in d.items():
   if name=='u':continue
   print(k,name,'LOO feature errors',[(lbl,round(cvfit(p,v),6)) for lbl,p in [('affine',A),('quadratic',Q),('quadratic+cube',C),('harmonic',H),('linear+cube',np.c_[A,u**3])]])
   for kind in ('tanh','sine','cubic','rational'):
    def pred(par,x):
     w=np.c_[np.ones(len(x)),x]@par[:20].reshape(5,4);a=np.exp(par[20:24]);b=par[24:];return b+a*({'tanh':np.tanh,'sine':np.sin,'cubic':lambda z:z+z**3,'rational':lambda z:z/np.sqrt(1+z*z)}[kind])(w)
    ini=np.r_[fit(A,v).ravel(),np.zeros(8)];p=least_squares(lambda p:(pred(p,u)-v).ravel(),ini,max_nfev=100).x
    err=np.sqrt(np.mean((pred(p,u)-v)**2));print(kind,round(err,6),np.round(np.exp(p[20:24]),3))
 def sgn(z):return np.sign(z)*np.sqrt(abs(z))
 basis=old._time_basis(S['eval_times'],tr['traj'][:52]); Y=np.c_[tr['y'],tr['traj']@basis]; print('time basis',basis.T)
 for zz,name in [(za,'inverse'),(np.r_[tr['x'],S['test']['x']],'observed')]:
  P,idx=poly(zz,3); b=fit(P[:52],Y[:52]); print(name,'dominant cubic terms',[[ (idx[j],round(b[j,k],4)) for j in np.argsort(abs(b[:,k]))[-12:][::-1]] for k in range(3)])
  for d in (1,2,3,4,5,6,7):
   X,_=poly(zz,d); X=X/X[:52].std(0).clip(.1); eye=np.eye(len(X[0]));eye[0,0]=0
   for p in (.00001,.001,.02,.1,1.):
    X0=X[:52]; XX=X0@X0.T; c=np.linalg.solve(XX+p*np.eye(52),Y[:52]); q=X[52:n]@X0.T@c; print(name,d,p,np.round(np.mean(abs(q-Y[52:]),axis=0),4),round(np.mean(abs(q[:,1:]@basis.T-tr['traj'][52:])),4))
