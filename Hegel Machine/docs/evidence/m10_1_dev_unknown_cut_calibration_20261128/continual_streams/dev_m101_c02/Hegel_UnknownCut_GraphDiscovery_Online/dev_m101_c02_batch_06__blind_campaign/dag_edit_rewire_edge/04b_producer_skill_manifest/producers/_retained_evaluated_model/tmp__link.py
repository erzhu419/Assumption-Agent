from explore import *
from itertools import combinations
C=(u*np.roll(bv,1,axis=1)).sum(1,keepdims=True)
B=bv
Z=tr
p0=np.arange(52); p1=np.arange(52,72)
rng=np.random.default_rng(184); splits=[(p0,p1)]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(6)]]
links={'lin':lambda x:x,'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'softsign':lambda x:x/np.sqrt(1+x*x),'rational':lambda x:x/(1+x*x),'id+cubic':lambda x:x-x**3/3}
for key,fn in links.items():
 for placement in ['outside','inside','both','separate','outside+u']:
  def pred(th,I):
   S=B[I]@th[:4]+th[4]; V=C[I,0]; H=th[5]*V if placement in ('inside','both') else 0
   if placement=='outside+u': S=S+u[I]@th[8:12]
   if placement=='separate': S=fn(B[I]*th[:4])@th[8:12]+th[4]
   else: S=th[6]*fn(S+H)
   return S+th[7]+(th[5]*V if placement in ('outside','outside+u','separate') else th[8]*V if placement=='both' else 0)
  start=np.r_[[-1,-1,1.3,.8],0,.7,1.,0,.5,.5,.5,.5]; start[8:12]=0 if placement=='outside+u' else np.sign(start[:4]) if placement=='separate' else .5
  rows=[]
  for I,J in splits:
   fit=least_squares(lambda th:pred(th,I)-y[I],start,max_nfev=650); rows.append([np.mean(abs(pred(fit.x,J)-y[J])),np.mean(abs(fit.fun))]);
   if J is p1: th1=fit.x.copy()
  print('y',key,placement,np.round(np.r_[rows[0],np.max(rows,0)[0]],6),'params',np.round(th1,3))
for key,fun in {'exp':lambda k:-np.expm1(-t*k),'log':lambda k:np.log1p(t*k),'tanh':lambda k:np.tanh(t*k),'rat':lambda k:t/(1+k*t),'atan':lambda k:np.arctan(t*k),'sin':lambda k:np.sin(t*k),'sqrt':lambda k:np.sqrt(1+k*t)-1,'expraw':lambda k:np.exp(-k*t)}.items():
 for second in ('same','exp','log','rat'):
  def design(k,I):
   g=fun(k[0]); h={'same':fun,'exp':lambda k:-np.expm1(-t*k),'log':lambda k:np.log1p(t*k),'rat':lambda k:t/(1+k*t)}[second](k[1]); return np.concatenate((B[I,:,None]*g,C[I,:,None]*h),1).transpose(0,2,1).reshape(-1,5)
  def fit(k,I):
   X=design(k,I); coef=np.linalg.lstsq(X,Z[I].ravel(),rcond=None)[0];return X@coef-Z[I].ravel(),coef
  errs=[]
  for I,J in splits:
   res=least_squares(lambda k:fit(k,I)[0],[.6,1.2],bounds=(.0001,8)); coef=fit(res.x,I)[1];errs.append(np.mean(abs(design(res.x,J)@coef-Z[J].ravel())))
   if J is p1:k1=res.x; c1=coef; rms=np.std(res.fun)
  print('traj',key,second,np.round([errs[0],max(errs),rms],6),np.round(k1,4),np.round(c1,4))
