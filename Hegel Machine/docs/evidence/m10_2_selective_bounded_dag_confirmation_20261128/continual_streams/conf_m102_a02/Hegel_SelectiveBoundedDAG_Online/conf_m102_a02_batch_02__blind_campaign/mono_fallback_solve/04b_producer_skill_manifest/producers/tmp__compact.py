import pickle,numpy as np
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
poly=lambda x,d:PolynomialFeatures(d).fit_transform(x)
p=s['panel_a']; u,v=p['u'],p['v']; m=np.linalg.norm(u-v,axis=1)<1
A=np.array([ls(poly(u[m,i:i+1],1),v[m,i]) for i in range(4)]); af=lambda x:x*A[:,1]+A[:,0]; ai=lambda x:(x-A[:,0])/A[:,1]
p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); bf=lambda x:poly(x,1)@B
from scipy.optimize import least_squares
p=s['panel_c']; u=p['u']; v=p['v1']; w=p['v2']; one=lambda q:np.ones((len(q),1)); aff=lambda q:np.c_[one(q),q]; Q=lambda q:poly(q,2)
fun={'softsign':lambda q,k:q/(1+k*abs(q)),'oddexp':lambda q,k:q*np.exp(-k*abs(q)),'soft':lambda q,k:q/np.sqrt(1+k*q*q),'tanh':lambda q,k:np.tanh(k*q),'asinh':lambda q,k:np.arcsinh(k*q),'cos':lambda q,k:np.cos(k*q),'logcosh':lambda q,k:np.logaddexp(k*q,-k*q)-np.log(2),'exp':lambda q,k:np.exp(-k*q*q),'abs':lambda q,k:abs(q),'odd2':lambda q,k:q*abs(q),'rat':lambda q,k:q/(1+k*q*q),'evenrat':lambda q,k:1/(1+k*q*q),'logabs':lambda q,k:np.sign(q)*np.log1p(k*abs(q))}
for n in ['softsign','oddexp','atan','logabs','cos','logcosh','exp']:
 f=fun.get(n,lambda q,k:np.arctan(k*q)); Y=w if n in ('cos','logcosh','exp') else v
 for lin in (False,True):
  D=lambda q,k:np.c_[aff(q) if lin else one(q),f(q,k)]
  opt=least_squares(lambda a:(D(u,*a)@ls(D(u,*a),Y)-Y).ravel(),[1.25],bounds=(.01,10),gtol=1e-11)
  print('panel',n,lin,opt.x,np.mean(abs(opt.fun)))
coords={'raw':x,'invA':ai(x),'A':af(x),'BA':bf(af(x)),'BinvA':bf(ai(x)),'Braw':bf(x)}
splits=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=12).split(x),*KFold(4,shuffle=True,random_state=73).split(x)]
def score(F):
 e=[np.mean(abs(F[b]@ls(F[a],y[a])-y[b])) for a,b in splits]; return tuple(np.round([e[0],np.mean(e[1:]),max(e[1:]),np.std(y-F@ls(F,y))],6))
results=[]
for nm,q in coords.items():
 L=aff(q); P=Q(q)
 for n,f in fun.items():
  for k in ([1] if n in ('abs','odd2') else [.22,.6,1.1,1.25,2,4,8]):
   z=f(q,k); ii=2
   for tag,F in [('all',np.c_[P,z]),('oneL',np.c_[P,z[:,ii,None]*L]),('oneQ',np.c_[P,z[:,ii,None]*P]),('oneDiag',np.c_[P,z[:,ii,None]*np.c_[L,q*q]]),('allL',np.c_[P,np.einsum('ni,nj->nij',z,L).reshape(len(q),-1)]),('directQ',Q(z)),('directL',np.c_[L,z])]:
    results.append((score(F),nm,n,k,tag))
for tag,key in [('worst',lambda r:max(r[0][0],r[0][2])),('train',lambda r:r[0][3])]: print(tag,*sorted(results,key=key)[:20],sep='\n')
q=ai(x); z=fun['soft'](q[:,2],4); F=np.c_[Q(q),z[:,None]*Q(q)]; b=ls(F,y); print('invA calibration',A); print('soft coeff',b.reshape(2,-1)); print('residual',np.sort(y-F@b)); print('ranges',q.min(0),q.max(0),ai(s['test']['x']).min(0),ai(s['test']['x']).max(0)); print('times',s['eval_times'])
