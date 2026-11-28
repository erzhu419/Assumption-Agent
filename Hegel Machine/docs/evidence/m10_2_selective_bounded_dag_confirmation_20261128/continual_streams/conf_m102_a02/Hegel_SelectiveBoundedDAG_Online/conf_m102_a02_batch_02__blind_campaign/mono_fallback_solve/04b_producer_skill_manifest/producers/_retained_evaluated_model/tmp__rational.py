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
from itertools import combinations
from scipy.linalg import qr
T=s['telemetry_vec']; print('telemetry',T)
splits=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=12).split(x)]
def score(F):
 e=[np.mean(abs(F[b]@ls(F[a],y[a])-y[b])) for a,b in splits]; return tuple(np.round([e[0],np.mean(e[1:]),max(e[1:])],5))
coords={'raw':x,'invA':ai(x),'A':af(x),'BA':bf(af(x)),'BinvA':bf(ai(x)),'Braw':bf(x)}
results=[]
for nm,q in coords.items():
 Q=poly(q,2); L=Q[:,:5]
 for i in range(4):
  for g in [.11,.22,.44,1.25]:
   d=1+g*q[:,i]**2
   for name,F in [('ratL',np.c_[Q,L[:,1:]/d[:,None]]),('ratQ',np.c_[Q,Q[:,1:]/d[:,None]]),('ratNum',Q/d[:,None]),('sqL',np.c_[Q,L[:,1:]*q[:,i,None]**2]),('sinL',np.c_[Q,L*np.sin(g*q[:,i,None])]),('tanOne',np.c_[Q,np.tanh(g*q[:,i])]),('cosL',np.c_[Q,L*np.cos(g*q[:,i,None])])]:
    results.append((score(F),nm,i,g,name,np.std(y-F@ls(F,y))))
print('rational worst'); print(*sorted(results,key=lambda r:max(r[0][0],r[0][2]))[:22],sep='\n')
print('rational split'); print(*sorted(results)[:15],sep='\n')
p=s['panel_c']; u=p['u']; csplit=list(KFold(6,shuffle=True,random_state=11).split(u)); fit=[]
for nm,fun in {'tanh':lambda q,k:np.tanh(k*q),'sin':lambda q,k:np.sin(k*q),'cos':lambda q,k:np.cos(k*q),'soft':lambda q,k:q/np.sqrt(1+k*q*q),'rat':lambda q,k:q/(1+k*q*q),'rat2':lambda q,k:q*q/(1+k*q*q),'sinabs':lambda q,k:np.sin(k*np.abs(q)),'abs':lambda q,k:np.abs(q),'expsq':lambda q,k:np.exp(-k*q*q),'logcosh':lambda q,k:np.logaddexp(k*q,-k*q),'tanh2':lambda q,k:np.tanh(k*q*q),'sign':lambda q,k:np.sign(q),'cubic':lambda q,k:q**3,'square':lambda q,k:q**2}.items():
 for k in [.22,.44,.75,1,1.25,1.5,2,2.5,3,4,5]:
  G=fun(u,k); F=np.c_[np.ones(len(u)),u,G]; F2=np.c_[F,u**2] if nm not in ('square',) else F
  for fname,H in [('base',F),('plusSquare',F2)]:
   beta=ls(H,np.c_[p['v1'],p['v2']]); tr=np.mean(abs(H@beta-np.c_[p['v1'],p['v2']]),axis=0).reshape(2,4).mean(1)
   V=np.c_[p['v1'],p['v2']]; cv=np.mean(np.concatenate([abs(H[b]@ls(H[a],V[a])-V[b]) for a,b in csplit]),axis=0).reshape(2,4).mean(1)
   fit.append((cv,tr,nm,k,fname))
for i in range(2):
 print('C',i,*sorted(fit,key=lambda r:r[0][i])[:15],sep='\n')
