import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def ls(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def rms(x):return np.sqrt(np.mean(x*x,axis=0))
u=s['panel_c']['u']; B=aff(u); P=PolynomialFeatures(2).fit_transform(u); R=np.eye(len(u))-B@np.linalg.pinv(B)
for k in ('v1','v2'):
 v=s['panel_c'][k]; print(k)
 for name,f,base in [('sin',np.sin,B),('tanh',np.tanh,B),('cos',np.cos,B),('quadratic+sin',np.sin,P),('quadratic+cos',np.cos,P),('quadratic+tanh',np.tanh,P)]:
  def X(z,p):return np.c_[base[z],f(u[z]*p)]
  jj=np.arange(24); aa=jj[:19]; bb=jj[19:]; best=None
  for w in (1.,2.,3.,4.,5.):
   def err(p):
    m=X(aa,p);return (m@ls(m,v[aa])-v[aa]).ravel()
   opt=least_squares(err,[w]*4,bounds=(.15,8),max_nfev=100); e=np.std(opt.fun)
   if best is None or e<best[0]: best=e,opt.x
  e,p=best; print(name,'p',p,'fit/test',e,np.std(X(bb,p)@ls(X(aa,p),v[aa])-v[bb]));
  if e<.007: print('coef',ls(X(aa,p),v[aa]))
x,y,q=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; a=s['panel_a']; cal=np.zeros_like(x)
for j in range(4):
 u,v=a['u'][:,j],a['v'][:,j];ii,jj=np.triu_indices(len(u),1);m=np.median((v[ii]-v[jj])/(u[ii]-u[jj]));d=np.median(v-m*u);ok=abs(v-d-m*u)<.02;d,m=ls(aff(u[ok,None]),v[ok]); cal[:,j]=(x[:,j]-d)/m
p=s['panel_b'];b=ls(aff(p['u']),p['v']); F=-np.expm1(-t[:,None]*[1.8131,.40838]); amp=q@np.linalg.pinv(F.T);Y=np.c_[y,amp]; print('polynomial coefficient sizes in calibrated basis')
for deg in (2,3):
 pf=PolynomialFeatures(deg);X=pf.fit_transform(cal); coef=ls(X,Y);print(deg,[(n,c) for n,c in zip(pf.get_feature_names_out(list('abcd')),coef)],rms(X@coef-Y))
for name,z in [('cal',cal),('B',aff(cal)@b),('Bi',(cal-b[0])@np.linalg.inv(b[1:]))]:
 print(name)
 for lab,make in [('add3',lambda z:np.c_[z,z*z,z**3]),('add5',lambda z:np.concatenate([z**d for d in range(1,6)],axis=1)),('pair3',lambda z:np.c_[z,z*z,z**3,*[z[:,i]*z[:,j] for i,j in it.combinations(range(4),2)]]),('poly3',lambda z:PolynomialFeatures(3,include_bias=False).fit_transform(z)),('tanh2',lambda z:np.c_[z,z*z,np.tanh(2*z),*[z[:,i]*z[:,j] for i,j in it.combinations(range(4),2)]]),('sin2',lambda z:np.c_[z,z*z,np.sin(2*z),*[z[:,i]*z[:,j] for i,j in it.combinations(range(4),2)]])]:
  X=make(z); pred=[]
  for a0 in [0,1,2,3,4]:
   va=(np.arange(72)%5==a0);tr=~va; reg=Ridge(alpha=.013).fit(X[tr],Y[tr]); p0=reg.predict(X[va]); pred.append([np.mean(abs(p0[:,0]-y[va])),np.max(np.mean(abs(p0[:,1:]@F.T-q[va]),axis=0))]);
  print(lab,'mean',np.mean(pred,0),'worst',np.max(pred,0))
