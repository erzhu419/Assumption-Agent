from model14 import *
from itertools import product,combinations_with_replacement
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; tr=s['train']; t=s['eval_times']; n=52
z=calibration(s['panel_a'])(np.r_[tr['x'],s['test']['x']]); print('z quantiles',np.quantile(z,[0,.01,.99,1]))
bp=s['panel_b']; b=D(z)@ls(D(bp['u']),bp['v']); ops=[]
for key in ('v1','v2'):
 v=p[key]; fits=[]
 for f,g,sg in product(('sin','cos','tanh','atan','sat','abs','soft','identity'),('cube','square','adj','opp','linear','triple'),(1,-1)):
  if g in ('cube','square','opp','linear') and sg<0:continue
  def extra(x):
   w=np.roll(x,sg,axis=1)
   return {'cube':x**3,'square':x*x,'adj':x*w,'opp':x*np.roll(x,2,axis=1),'linear':x,'triple':x*w*np.roll(x,2,axis=1)}[g]
  F=fun.get(f,lambda a:a)
  for mode in ('shared','separate','linshared'):
   def design(x,k):
    w=extra(x); a=F(k[0]*x)
    return D(a+k[1]*w) if mode=='shared' else D(np.c_[x,a+k[1]*w]) if mode=='linshared' else D(np.c_[a,w])
   def resid(k):
    a=design(u,k);return (a@ls(a,v)-v).ravel()
   for k0 in (.85,1.85,3.5):
    fit=least_squares(resid,[k0,.22],bounds=([.08,-12],[6,12]),max_nfev=55); a=design(u,fit.x); err=np.std(fit.fun.reshape(v.shape)/(1-np.diag(a@np.linalg.pinv(a)))[:,None]); fits.append((err,f,g,sg,mode,fit.x,design(z,fit.x)@ls(a,v)))
 fits.sort(key=lambda a:a[0]); seen=set();print(key)
 for q in fits:
  if q[1:5] not in seen:
   print(q[:6]);seen.add(q[1:5])
  if len(seen)==9:break
 ops.append(fits[0][-1]);
p,q=ops
T=tr['traj']; basis=lambda rates:-np.expm1(-t[:,None]*rates); rates=least_squares(lambda a:(basis(a)@ls(basis(a),T[:n].T)-T[:n].T).ravel(),[1.85,.4]).x; H=basis(rates); Y=np.c_[tr['y'],ls(H,T.T).T]; print('rates and target scales',rates,Y.std(0));
F={}
def add(name,a): F[name]=np.asarray(a).reshape(len(z),-1)
for k,a in dict(z=z,b=b,p=p,q=q,pplusq=p+q,pminusq=p-q,pq=p*q).items():
 add(k,a)
 for f,fn in fun.items():
  for t0 in (.22,1,2.15):add(f'{f}({t0}{k})',fn(t0*a))
for (k,a),(l,d) in combinations_with_replacement(dict(z=z,b=b,p=p,q=q).items(),2):
 add(k+l,a*d);add(k+'x'+l,a[:,:,None]*d[:,None,:]);add(k+'dot'+l,(a*d).sum(1));add(k+'rat'+l,a/(1+.22*d*d));add(l+'rat'+k,d/(1+.22*a*a));add('sin('+k+l+')',np.sin(2.15*a*d));add('tanh('+k+l+')',np.tanh(2.15*a*d))
def score(A,Y):
 a=A[:n]; bh=ls(a,Y[:n]); h=np.sum(a*np.linalg.pinv(a).T,axis=1); e=(a@bh-Y[:n])/(1-h.clip(max=.999)); return np.mean(e**2)**.5,np.mean(abs(A[n:72]@bh-Y[n:])),np.mean(abs(a@bh-Y[:n]))
for i,y in enumerate(Y.T):
 print('TARGET',i,'best single',sorted((score(D(a),y),k) for k,a in F.items())[:9]); keys=list(F); A=np.ones((len(z),1)); sel=[]
 for j in range(8):
  k=min(keys,key=lambda k:score(np.c_[A,F[k]],y)[0]);A=np.c_[A,F[k]];sel.append(k);keys.remove(k); print(j,sel,score(A,y))
np.savez('/tmp/graph14.npz',z=z,b=b,p=p,q=q,Y=Y,H=H)
