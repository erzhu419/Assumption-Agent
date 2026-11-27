from rheology_base import *
from scipy.optimize import least_squares
P,H,X=pq; D=X@b[2:]; Y=np.column_stack((q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs'])); W=np.r_[np.ones(len(t)),1.]
def temporal(z):return np.vstack((np.column_stack((-np.expm1(-t[:,None]/z),t)),np.r_[1/z,1.]))
def mem(tt,rev=False):
 m=np.zeros((len(H),2)); ii=range(len(dt)-1,-1,-1) if rev else range(len(dt))
 for s in ii:m+=-np.expm1(-dt[s]/tt)*(P[:,s,None]-m)
 return m
M=mem(tau); R=mem(tau,True)
F={'1':np.ones(len(H)),'hold':H,'drift':D,'drift2':D**2,'drift3':D**3, 'clipD':np.maximum(D,0),'sqrtH':np.sqrt(np.abs(H)), 'H2':H**2,'H3':H**3,'tanhH':np.tanh(H)}
for j in range(2):
 for name,v in [('M',M[:,j]),('R',R[:,j]),('gap',H-M[:,j]),('rgap',H-R[:,j])]: F[name+str(j)]=v; F['sq_'+name+str(j)]=v*v;F['H*'+name+str(j)]=H*v;F['abs_'+name+str(j)]=np.abs(v)
for j in range(P.shape[1]):
 F[f'p{j}']=P[:,j];F[f'p{j}2']=P[:,j]**2;F[f'tanhp{j}']=np.tanh(P[:,j]);F[f'dh{j}']=H-P[:,j]
for j in range(1,X.shape[1]):F[f'x{j}']=X[:,j]
for j in range(4):
 F[f'hold_ch{j}']=q['hold_channel_raw'][:,j]
 for k in range(P.shape[1]):F[f'past_ch{k}_{j}']=q['past_channel_raw'][:,k,j]
for fit_tau in [False,True]:
 z=np.exp(least_squares(lambda l:((np.eye(len(W))-temporal(np.exp(l))@np.linalg.pinv(temporal(np.exp(l))))@Y.T).ravel(),np.log(tau),max_nfev=100).x) if fit_tau else tau
 A=temporal(z); C=np.linalg.lstsq(A,Y.T,rcond=None)[0].T; print('time constants',z,'RMSE',np.sqrt(np.mean((C@A.T-Y)**2,axis=0))[[0,5,11,12]])
 for j in range(3):
  names=['1']; yy=C[:,j]; print('amplitude',j,'std',np.std(yy)); print('top correlations',sorted([(abs(np.corrcoef(v,yy)[0,1]),n) for n,v in F.items() if n!='1'],reverse=True)[:10]);
  for k in range(4):
   def score(n):
    f=np.column_stack([F[s] for s in names+[n]]); Q=np.linalg.qr(f)[0]; e=(yy-Q@(Q.T@yy))/(1-np.sum(Q*Q,axis=1));return np.sqrt(np.mean(e*e))
   sc,n=min((score(n),n) for n in F if n not in names and np.linalg.matrix_rank(np.column_stack([F[s] for s in names+[n]]))==len(names)+1); names.append(n); cf=np.linalg.lstsq(np.column_stack([F[s] for s in names]),yy,rcond=None)[0];print('  LOO',round(sc,5),list(zip(names,np.round(cf,4))))
 print('amplitudes/drift ',np.round(np.column_stack((C,D,H,P[:,[-2,-1]])),4))
