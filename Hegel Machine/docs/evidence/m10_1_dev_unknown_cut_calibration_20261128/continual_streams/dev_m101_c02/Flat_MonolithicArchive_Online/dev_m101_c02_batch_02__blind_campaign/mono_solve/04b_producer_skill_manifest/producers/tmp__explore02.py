import pickle,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; a=s['panel_a']; b=s['panel_b']; keep=np.max(abs(a['v']-a['u']),axis=1)<1
A=lambda z:np.c_[np.ones(len(z)),z]
cal=np.array([np.linalg.lstsq(A(a['u'][keep,j:j+1]),a['v'][keep,j],rcond=None)[0] for j in range(4)])
w=(x-cal[:,0])/cal[:,1]; op=np.linalg.lstsq(A(b['u']),b['v'],rcond=None)[0]; z=A(w)@op
pairs=list(itertools.combinations_with_replacement(range(4),2))
Q=lambda u:np.column_stack([A(u)]+[u[:,i]*u[:,j] for i,j in pairs]); d=Q(z)
tr=np.arange(56); va=np.arange(56,72)
p=least_squares(lambda p:(p[0]+p[1]*np.arctan(d@p[2:])-y)[tr],np.r_[0,2,np.linalg.lstsq(d[tr],y[tr],rcond=None)[0]/2]).x
h=d@p[2:]; f=p[0]+p[1]*np.arctan(h); e=y-f
print('cal',cal,'scalar arctan coefficients',p.round(5)); print('raw congruent coefficients',np.linalg.lstsq(Q(w),h,rcond=None)[0].round(5)); print('largest residuals: y, pred, latent, e, z'); ix=np.argsort(abs(e))[-18:][::-1]; print(np.c_[np.arange(len(y)),y,f,h,e,z][ix].round(4)); print('upper/lower tails');ix=np.r_[np.argsort(y)[:8],np.argsort(y)[-12:]];print(np.c_[np.arange(len(y)),y,f,h,e,z][ix].round(4));
F=A(z); C=np.linalg.lstsq(F,T,rcond=None)[0]; print('trajectory coefficients in B coords\n',C.T.round(5));
for name,basis,pi,lb,ub in [('log1p',lambda p: np.log1p(t[:,None]*p),[1.],[.001],[20]),('exp',lambda p:-np.expm1(-t[:,None]*p),[1.],[.001],[20]),('rational',lambda p:t[:,None]/(1+t[:,None]*p),[1.],[.001],[20]),('log+exp',lambda p:np.c_[np.log1p(t*p[0]),-np.expm1(-t*p[1])],[1,1],[.001,.001],[20,20]),('2exp',lambda p:-np.expm1(-t[:,None]*p),[.4,2],[.001,.001],[20,20]),('2log',lambda p:np.log1p(t[:,None]*p),[.4,2],[.001,.001],[20,20])]:
 def calc(p,i):
  B=basis(p); return (F[i]@np.linalg.lstsq(F[tr],np.linalg.lstsq(B,T[tr].T,rcond=None)[0].T,rcond=None)[0])@B.T
 opt=least_squares(lambda p:(calc(p,tr)-T[tr]).ravel(),pi,bounds=(lb,ub),max_nfev=60).x
 print('traj',name,opt.round(5),np.abs(calc(opt,tr)-T[tr]).mean(),np.abs(calc(opt,va)-T[va]).mean());
