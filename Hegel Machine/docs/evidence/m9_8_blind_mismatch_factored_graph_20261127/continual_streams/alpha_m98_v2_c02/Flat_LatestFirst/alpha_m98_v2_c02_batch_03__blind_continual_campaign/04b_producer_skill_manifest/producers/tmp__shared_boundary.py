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
# Infer the common temporal signature without allowing a separate amplitude for each branch.
sv=np.linalg.svd(E(a),full_matrices=False); Cdir=np.linalg.svd(C,full_matrices=False)[2][0]; sig=F@Cdir; yc=(E(a)@sig)/(sig@sig)
print('common direction',Cdir,'amp',yc.round(4)); print('diag vs bench std',np.std(y,0).round(3))
J={'r':r,'H':H}; J.update({f'x{i}':u for i,u in enumerate(x.T)})
for name,A in [('P',P),('dP',np.diff(np.c_[np.zeros(len(P)),P,H],axis=1)),('U',U.reshape(len(P),-1)),('V',V),('P-H',P-H[:,None]),('z',Z),('Umean',U.mean(1)),('Ulast',U[:,-1]),('Ustd',U.std(1)),('Upastmax',U.max(1)),('Upastmin',U.min(1)),('Uhold-last',V-U[:,-1]),('Ustdchannel',U.std(2)),('Uminchannel',U.min(2)),('Umaxchannel',U.max(2))]:
 J.update({name+str(k):u for k,u in enumerate(A.T)})
for name,A in [('P',P),('dP',np.diff(np.c_[np.zeros(len(P)),P,H],axis=1)),('x',x),('V',V)]:
 for suffix,fun in [('avg',np.mean),('min',np.min),('max',np.max),('std',np.std),('rng',np.ptp)]: J[name+suffix]=fun(A,axis=1)
B={}; add=lambda k,u:B.update({k:u})
for i,(k,u) in enumerate(J.items()):
 add(k,u); add(k+'²',u*u)
 for l,w in list(J.items())[:i]:
  add(k+'*'+l,u*w)
  if k.startswith(('x','V','P')) and l.startswith(('x','V','P')): add(k+'+'+l,u+w); add(k+'-'+l,u-w)
labels=list(B); Mat=np.column_stack(list(B.values())); valid=Mat.std(0)>1e-12; Mat=Mat[:,valid]; labels=np.array(labels)[valid]
yy=-yr; n=len(yy)
def select(train):
 m=Mat[train]; target=yy[train]; mu=m.mean(0); sd=np.maximum(m.std(0),1e-9); A=(m-mu)/sd; cen=target.mean(); z=target-cen; R=[]
 for kind,pred in [('linear',A),('square',A*A)]+[(f'hinge{t:+.2f}',np.maximum(A-t,0)) for t in (-1.,-.5,0.,.5,1.)]+[(f'hinge{t:+.2f}-',np.maximum(-A-t,0)) for t in (-1.,-.5,0.,.5,1.)]:
  v=pred-pred.mean(0); den=np.maximum((v*v).sum(0),1e-12); c=(z@v)/den; e=z[:,None]-v*c; rms=np.mean((e/np.maximum(1-1/len(target)-v*v/den,.03))**2,0)**.5
  for j in np.argsort(rms)[:8]: R.append((rms[j],labels[j],kind,c[j],j,mu[j],sd[j],cen-c[j]*pred[:,j].mean()))
 return sorted(R)
rows=select(np.arange(n)); print('boundary screen',[(round(err,4),name,k,round(c,4)) for err,name,k,c,*_ in rows[:24]])
print('nested winners',[(i,select(np.delete(np.arange(n),i))[0][1:3]) for i in (0,4,6,8,13,16)])
# Check whether drift has a residual proportional to its intrinsic coefficient after removing memory.
for name,zz in [('integrated',lambda d:T+d*np.expm1(-T/d)),('saturating',lambda d:T+d*np.expm1(-T/d))]:
 Q1=np.eye(len(T))-np.outer(sig,sig)/(sig@sig)
 fit=least_squares(lambda p:((Y+p[0]*r[:,None]*zz(p[1]))@Q1).ravel(),[1.,1.],bounds=([.3,.1],[2.,4.])).x
 e=(Y+fit[0]*r[:,None]*zz(fit[1]))@Q1; print(name,fit,'per-time orth error',np.sqrt(np.mean(e**2,axis=0)).round(4)); break
