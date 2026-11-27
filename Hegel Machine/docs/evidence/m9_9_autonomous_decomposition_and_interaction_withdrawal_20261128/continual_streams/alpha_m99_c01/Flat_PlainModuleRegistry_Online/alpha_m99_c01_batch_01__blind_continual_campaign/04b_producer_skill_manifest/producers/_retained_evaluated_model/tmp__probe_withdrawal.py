exec(open('/tmp/probe_chamber.py').read().split('for tag,')[0])
from scipy.optimize import least_squares
Y=np.c_[yc,rc]; n,k=Y.shape
age=np.cumsum(dt[::-1])[::-1]-dt
z=lambda pp,u: pp@(np.exp(-age[:,None]/u)*-np.expm1(-dt[:,None]/u))
def basis(u): return np.r_[-np.expm1(-t[:,None]/u),[1/u]]
a,b=base._design(tau,xc,pc,hc,dt,t)
nom=b[:,:2]*tau*beta[:2]; drift=b[:,2:]@beta[2:]
for kind in ('zero','global','nominal','scaled','free'):
 def fit(logu):
  u=np.exp(logu); T=basis(u); q=np.r_[t,1.]; F=np.kron(np.eye(n),T); target=Y.copy()
  if kind=='global': F=np.c_[F,np.tile(q,n)]
  if kind=='nominal': target-=drift[:,None]*q
  if kind=='scaled': F=np.c_[F,(drift[:,None]*q).ravel()]
  if kind=='free': F=np.kron(np.eye(n),np.c_[T,q]);
  coef=np.linalg.lstsq(F,target.ravel(),rcond=None)[0]; return (F@coef-target.ravel()).reshape(n,k),coef
 sol=least_squares(lambda v:fit(v)[0].ravel(),np.log(tau),max_nfev=90)
 res,coef=fit(sol.x); print('drift',kind,'taus',np.exp(sol.x),'RMSE curve/rate',np.sqrt(np.mean(res[:,:-1]**2)),np.sqrt(np.mean(res[:,-1]**2)),'extra',coef[-1] if kind in ('global','scaled') else '')
 if kind=='scaled': best_u=np.exp(sol.x); amp=coef[:2*n].reshape(n,2)
print('lab/hold/cal state, inferred branch amplitudes')
print(np.round(np.c_[d['lab_id'],hc,z(pc,best_u),pc[:,-2:],amp],4))
for j in range(2):
 state=z(pc,best_u); gj=hc-state[:,j];
 for name,A in [('gap',gj[:,None]),('hold+state',np.c_[hc,state]),('hold+history',np.c_[hc,pc]),('gap+drift',np.c_[gj,drift]),('hold+history+drift',np.c_[hc,pc,drift]),('hold+history+features',np.c_[hc,pc,xc]),('hold+history+lab',np.c_[hc,pc,np.eye(4)[d['lab_id']][:,1:]])]:
  A=np.c_[np.ones(n),A]; c=np.linalg.lstsq(A,amp[:,j],rcond=None)[0]; e=A@c-amp[:,j]; lev=np.sum(A*np.linalg.pinv(A).T,axis=1); print(j,name,'LOO RMSE',round(np.sqrt(np.mean((e/(1-lev))**2)),6),'coeff',np.round(c,4))
