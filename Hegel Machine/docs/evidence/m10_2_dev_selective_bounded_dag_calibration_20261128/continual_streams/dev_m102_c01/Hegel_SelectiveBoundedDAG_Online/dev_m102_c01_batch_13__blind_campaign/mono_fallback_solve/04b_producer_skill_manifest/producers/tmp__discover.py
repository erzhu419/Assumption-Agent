import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; ts=s['eval_times']; n=52
fit=lambda A,Y:np.linalg.lstsq(A,Y,rcond=None)[0]
ones=lambda A:np.c_[np.ones(len(A)),A]
A,B,C=(s['panel_'+k] for k in 'abc'); u,v=A['u'],A['v']; use=np.linalg.norm(v-u,axis=1)<4; scale,off=np.array([fit(ones(u[use,j,None]),v[use,j]) for j in range(4)]).T[::-1]; b=fit(ones(B['u']),B['v'])
acts={'tanh':lambda x,k:np.tanh(k*x),'atan':lambda x,k:np.arctan(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'sqrt':lambda x,k:x/np.sqrt(1+k*x*x),'softsign':lambda x,k:x/(1+k*abs(x)),'erf':lambda x,k:erf(k*x),'rat':lambda x,k:x/(1+k*x*x),'sin':lambda x,k:np.sin(k*x),'rat2':lambda x,k:x*x/(1+k*x*x),'exp':lambda x,k:np.exp(k*x)}
for name,f in acts.items():
 for add in (0,1):
  def des(x,k):return ones(np.c_[x,f(x,k)] if add else f(x,k))
  def err(k):
   F=des(C['u'],k); Q=np.linalg.qr(F)[0]; E=C['v1']-Q@(Q.T@C['v1']); return np.mean((E/(1-(Q*Q).sum(1)[:,None]))**2)**.5
  r=minimize_scalar(err,bounds=(.001,7),method='bounded'); print(name,add,r.x,r.fun)
  if name in ['atan','sqrt','softsign','tanh'] and not add:print(fit(des(C['u'],r.x),C['v1']))
w1=fit(ones(acts['sqrt'](C['u'],1.)),C['v1']); f2=lambda z:ones(np.c_[z,acts['rat2'](z,.16185)]); w2=fit(f2(C['u']),C['v2']); print('W2',w2)
for kind,prof in [('exp',lambda k:-np.expm1(-k*ts)),('rat',lambda k:ts/(1+k*ts)),('power',lambda k:ts**k),('tanh',lambda k:np.tanh(k*ts))]:
 def err(k):
  h=prof(k); return np.mean((T-(T@h/np.dot(h,h))[:,None]*h)**2)**.5
 r=minimize_scalar(err,bounds=(.01,4),method='bounded'); print('profile',kind,r.x,r.fun)
h=-np.expm1(-.95*ts); a=T@h/(h@h)
x=tr['x']; z=(x-off)/scale; zz=ones(z)@b
J,K=np.triu_indices(4)
for name,q in [('x',x),('z',z),('b',zz),('C1(z)',ones(acts['sqrt'](z,1))@w1),('C1(b)',ones(acts['sqrt'](zz,1))@w1),('C2(z)',f2(z)@w2),('C2(b)',f2(zz)@w2)]:
 mats=[('linear',ones(q)),('quadratic',ones(np.c_[q,q[:,J]*q[:,K]]))]; mats += [(m+str(k),ones(np.c_[q,f(q,k)])) for m,f in acts.items() for k in [.22,1.25]]
 scores=[]
 for key,F in mats:
  P=F[n:]@fit(F[:n],np.c_[y,a][:n]); scores.append((key,*np.mean(abs(P-np.c_[y,a][n:]),axis=0)))
 print(name,sorted(scores,key=lambda v:v[1])[:4],sorted(scores,key=lambda v:v[2])[:4])
