import glob, importlib.util, json, pickle
import numpy as np
p=glob.glob('/tmp/hegel_continual_archive/*batch_02_solver.py')[0]
s=importlib.util.spec_from_file_location('base',p); b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']
xtr,xte=b._standardize(tr,te); stress=b._stress_calibration(cal)
a=b._design(tr,xtr,stress,dt,t); tau,beta=b._fit(a,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('dt',dt,'times',t,'tau',tau,'beta',beta)
from scipy.optimize import least_squares
from scipy.special import expit
x=b._standardize({k:np.concatenate((tr[k],te[k])) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']},q)[1]
S,R=b._design(q,x,stress,dt,t)(tau)
y=q['strain_curves_obs']-q['strain_0'][:,None]-S@beta; yr=q['instant_rate_obs']-R@beta
f=-np.expm1(-t[:,None]/tau); Y=np.column_stack((y,yr)); T=np.r_[t,0.]; F=np.column_stack((f.T,1/tau)).T
from scipy.optimize import minimize_scalar
P=stress(q['past_channel_raw']); H=stress(q['hold_channel_raw']); tail=np.cumsum(dt[::-1])[::-1]-dt
W=lambda tt: -np.expm1(-dt[:,None]/np.atleast_1d(tt))*np.exp(-tail[:,None]/np.atleast_1d(tt))
r=beta[2]+x@beta[3:]
G=lambda a:T+a*np.expm1(-T/a)
E=lambda a:(Y+r[:,None]*G(a))
Q=np.eye(len(T))-F@np.linalg.pinv(F)
a=minimize_scalar(lambda a:np.sum((E(a)@Q)**2),bounds=(.1,4),method='bounded').x
C=E(a)@np.linalg.pinv(F).T
Z=P@W(tau); Zc=Z-C/beta[:2]
raw=cal['bridge_raw']; L=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],cal['true_channel_loads'],rcond=None)[0]; U=q['past_channel_raw']@L[:4]+L[4]; V=q['hold_channel_raw']@L[:4]+L[4]
from scipy.special import expit
from scipy.optimize import nnls
sig=np.linalg.svd(E(a),full_matrices=False)[2][0]; sig*=np.sign(sig[-1]); amp=-(E(a)@sig); target=amp
np.set_printoptions(precision=3,suppress=True,linewidth=155)
print('i amp V0..3 V3-V0 Pstd x0..3'); print(np.c_[np.arange(len(P)),amp,V,V[:,3]-V[:,0],P.std(1),x]); print('sig',sig)
Lags=np.c_[np.zeros(len(P)),P,H]; D=np.diff(Lags,axis=1)
X={'V0':-V[:,0], 'V3-V0':V[:,3]-V[:,0], 'Vrange':np.ptp(V,axis=1), 'Vstd':V.std(1), 'V/H':-V[:,0]/H,'minV':-V.min(1),'Pstd+V0':-P.std(1)-V[:,0], 'Pstd*V0':-P.std(1)*V[:,0], 'Prange*V0':-np.ptp(P,axis=1)*V[:,0], 'x0':-x[:,0], 'xmean':-x.mean(1), 'r':-r}
def fit(z,m,j):
 pred=[]
 for power in (1,2):
  def err(p): return p[0]*np.maximum(z[m]-p[1],0)**power-target[m]
  best=min((least_squares(err,[1.,s],bounds=([0,-4],[30,3])) for s in (-1,0,0.5,1)),key=lambda f:f.cost)
  p=best.x; pred.append((best.cost, p[0]*np.maximum(z[j]-p[1],0)**power,p,power))
 return min(pred,key=lambda c:c[0])
for k,u in X.items():
 z=(u-u.mean())/u.std(); n=len(z); p=np.array([fit(z,np.arange(n)!=j,j)[1] for j in range(n)]); a0=fit(z,np.ones(n,bool),0); print(k,'rmse/max',np.sqrt(np.mean((p-target)**2)),max(abs(p-target)),'in',a0[0],a0[2:]);
# Test constant-drift gate with a shared amplitude from monotone saturating single-index models.
for name,Z in [('hold',V),('hold+hist',np.c_[V,P.std(1),np.ptp(P,axis=1)]),('features',x)]:
 mu=Z.mean(0); sd=Z.std(0); Z=(Z-mu)/sd
 for reg in (.04,.12):
  def f(m):
   def err(p): return np.r_[p[0]*np.logaddexp(0,Z[m]@p[2:]+p[1])-target[m],reg*p[2:]]
   return least_squares(err,np.r_[.1,-1,np.zeros(Z.shape[1])],bounds=(np.r_[0,-10,np.full(Z.shape[1],-8)],np.r_[1,2,np.full(Z.shape[1],8)])).x
  n=len(Z); p=np.stack([f(np.arange(n)!=j) for j in range(n)]); yh=p[:,0]*np.logaddexp(0,np.sum(Z*p[:,2:],1)+p[:,1]); print(name,reg,np.sqrt(np.mean((yh-target)**2)),max(abs(yh-target)),f(np.ones(n,bool)))
