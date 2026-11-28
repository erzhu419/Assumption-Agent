exec(open('/tmp/probe.py').read().split("for name,u,x in")[0])
from scipy.optimize import least_squares
from itertools import combinations
p,h=_loads(ca,q); y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; Y=np.c_[y,r*.35]; labs=q['lab_id']
def timefit(tau):
 A=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t], .35*np.r_[1/tau,1]])
 b=np.linalg.lstsq(A,Y.T,rcond=None)[0].T
 return A,b,b@A.T-Y
for k in (2,3):
 f=least_squares(lambda v: timefit(np.exp(v))[2].ravel(),np.log([.55,3.4] if k==2 else [.2,.55,3.4]),bounds=(np.log(.051),np.log(24.9)),max_nfev=400); u=np.exp(f.x); print('profile',u,summary(timefit(u)[2]),'rank SVD',np.linalg.svd(Y,compute_uv=False)[:6]);
 if k==2: ut=u
for u in (tau,ut):
 A,coef,e=timefit(u); print('fixed',u,'timefit',summary(e));
 Z=np.zeros((len(h),2));
 for j,w in enumerate(dt): Z=np.exp(-w/u)*Z-np.expm1(-w/u)*p[:,j,None]
 one=np.ones(len(h)); D=xq@th[2:]; L=np.eye(4)[labs][:,1:]; S=np.eye(3)[q['stratum']][:,1:]; print('coef std',coef.std(0));
 for name,F in [('H',np.c_[one,h]),('HZ',np.c_[one,h,Z]),('HP',np.c_[one,h,p]),('HPX',np.c_[one,h,p,xq[:,1:]]),('HZD',np.c_[one,h,Z,D]),('HZDL',np.c_[one,h,Z,D,L]),('HZDS',np.c_[one,h,Z,D,S]),('HZQ',np.c_[one,h,Z,h*h,h[:,None]*Z,Z*Z,Z.prod(1)]),('HX',np.c_[one,h,xq[:,1:]]),('samesign',np.c_[h[:,None]-Z,D]),('diff',np.c_[one,h,Z,Z[:,0]-Z[:,1],D])]:
  W=np.linalg.lstsq(F,coef,rcond=None)[0]; eh=F@W-coef; ep=np.zeros_like(coef);
  for lab in set(labs):
   j=labs==lab; ep[j]=F[j]@np.linalg.lstsq(F[~j],coef[~j],rcond=None)[0]-coef[j]
  cv=ep@A.T+e; print(name,'coef rmse', np.round(np.mean(eh**2,0)**.5,5),'lolo curve/rate',np.sqrt(np.mean(cv[:,:-1]**2)),np.sqrt(np.mean(cv[:,-1]**2))/.35, 'weights',W if name in ('HZD','HP') else '')
