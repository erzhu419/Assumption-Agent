import pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; y=p['v2']; O=np.ones((len(u),1)); X=np.c_[O,u]; q,r=np.linalg.qr(X); Y=y-q@(q.T@y)
def score(B):
 B=B-q@(q.T@B); Q,R=np.linalg.qr(B); R=Y-Q@(Q.T@Y); loo=R/np.maximum(1-(q*q).sum(1)-(Q*Q).sum(1),1e-8)[:,None];return np.sqrt(np.mean(R**2)),np.sqrt(np.mean(loo**2))
print('range',np.max(abs(u)), 'linear',np.sqrt(np.mean(Y**2)))
for name,B in [('squares',u**2),('cross',np.array([u[:,i]*u[:,j] for i,j in combinations(range(4),2)]).T),('cubic',u**3),('squares+cubic',np.c_[u**2,u**3]),('quartic',np.c_[u**2,u**4]),('abs',abs(u))]: print(name,score(B))
for name,f in [('rat2',lambda a:u**2/(1+a*u**2)),('rat4',lambda a:u**2/(1+a*u**4)),('sin',lambda a:np.sin(a*u)),('cos',lambda a:np.cos(a*u)),('tanh',lambda a:np.tanh(a*u)),('atan',lambda a:np.arctan(a*u)),('exp',lambda a:np.exp(-a*u**2)),('exp_lin',lambda a:np.exp(a*u)),('softabs',lambda a:np.sqrt(1+a*u**2)),('log',lambda a:np.log1p(a*u**2)),('softsign',lambda a:u/(1+a*abs(u))),('ratodd',lambda a:u/(1+a*u**2)),('ratabs',lambda a:u**2/(1+a*abs(u))),('power',lambda a:abs(u)**a)]:
 for sq in [False,True]:
  g=lambda a:score(np.c_[u**2,f(a)] if sq else f(a))
  grid=np.geomspace(.025,12,90); i=np.argmin([g(x)[0] for x in grid]); v=minimize_scalar(lambda x:g(x)[0],bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded'); print(name,sq,round(v.x,6),g(v.x), 'at .22/2.15',g(.22),g(2.15))
