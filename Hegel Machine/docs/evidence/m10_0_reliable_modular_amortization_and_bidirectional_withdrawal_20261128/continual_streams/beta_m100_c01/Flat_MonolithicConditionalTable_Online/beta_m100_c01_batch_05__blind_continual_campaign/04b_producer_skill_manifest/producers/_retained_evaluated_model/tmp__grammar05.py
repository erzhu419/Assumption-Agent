import numpy as np
from scipy.optimize import least_squares
exec(open('/tmp/fit_rosette05.py').read().split('for j in range(4):')[0])
P=np.array([own_fit(j)[0] for j in range(4)]); q=X; o=np.roll(X,-2,axis=1); yn=(Y-P[:,0])/P[:,1]; e=yn-q-.25*np.sin(2.5*q); ob=o[good].ravel(); qb=q[good].ravel(); eb=e[good].ravel(); ax=np.repeat(axis[good],4); valid=ax&(ob!=0); print('opp normalized',np.unique(np.c_[ob[valid],eb[valid].round(5)],axis=0));
F={
'opp phase':(lambda q,o,k:[o,np.sin(k[0]*o),np.cos(k[0]*o)-1],[2.5]),
'opp quad+sine':(lambda q,o,k:[o,o**2,np.sin(k[0]*o)],[2.5]),
'opp rational phase':(lambda q,o,k:[o,(o+o**2*k[1])/(1+k[0]*o**2)],[1.,-.3]),
'opp rational shift':(lambda q,o,k:[o,(o+k[1])/(1+k[0]*(o+k[1])**2)-k[1]/(1+k[0]*k[1]**2)],[1.,-.3]),
'mixed same':(lambda q,o,k:[o,o**2,np.sin(2.5*(q+k[0]*o))-np.sin(2.5*q)],[.2]),
'mixed diff':(lambda q,o,k:[o,o**2,np.sin(k[1]*(q+k[0]*o))-np.sin(k[1]*q)],[.2,2.5]),
'opp exp':(lambda q,o,k:[o,np.expm1(k[0]*o)],[.5]),
'opp tanh phase':(lambda q,o,k:[o,np.tanh(k[0]*o+k[1])-np.tanh(k[1])],[1,.3]),
'opp poly':(lambda q,o,k:[o,o**2,o**3,o**4,o**5,o**6],[1.]),
'opp full phase':(lambda q,o,k:[o,o**2,np.sin(k[0]*o),np.cos(k[0]*o)-1],[2.5]),
}
for name,(f,k0) in F.items():
 def fit(train,test):
  def residual(k):
   A=np.column_stack(f(qb,ob,k)); p=np.linalg.lstsq(A[train],eb[train],rcond=1e-10)[0]; return eb-A@p,p
  sol=least_squares(lambda k:residual(k)[0][train],k0,max_nfev=300,gtol=1e-12); err,p=residual(sol.x); return np.sqrt(np.mean(err[test]**2)),sol.x,p,err
 rms,k,p,er=fit(np.ones(len(eb),bool),np.ones(len(eb),bool)); ii=np.repeat(np.arange(sum(good)),4); fs=[fit(ii%5!=i,ii%5==i)[0] for i in range(5)]; print(name,'rmse',rms,'worst',max(fs),'k',k,'p',p)
 if name=='opp poly':
  m=~ax
  print('remaining dependence',np.linalg.lstsq(np.c_[qb[m]*ob[m], qb[m]**2*ob[m], qb[m]*ob[m]**2, np.sin(2.5*qb[m])*ob[m],(np.cos(2.5*qb[m])-1)*ob[m]],er[m],rcond=None)[0]); print('residual by channel',np.sqrt(np.mean(er.reshape(-1,4)**2,axis=0)))
print('opposite shape near zero and beyond', np.c_[ob[np.argsort(abs(ob))[:15]], eb[np.argsort(abs(ob))[:15]]]);print('mixed candidates axis b/c',np.c_[qb[~ax][:15],ob[~ax][:15],eb[~ax][:15]])
