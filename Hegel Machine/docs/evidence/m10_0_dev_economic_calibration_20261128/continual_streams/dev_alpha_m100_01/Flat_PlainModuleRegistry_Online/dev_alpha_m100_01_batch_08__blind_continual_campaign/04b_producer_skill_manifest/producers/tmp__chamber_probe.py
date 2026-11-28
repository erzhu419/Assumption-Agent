from chamber_data import *
from itertools import combinations
from scipy.optimize import least_squares
z=h[:,None]-g; l=di['lab_id']; st=di['stratum']; one=np.ones(len(h)); base=c.copy(); base[:,0]=th[0]*h
Y=y-base@B.T; a=B[:,0]; b1=B[:,1]; d=B[:,2]
Q={'hold':h,'zfast':z[:,0],'zslow':z[:,1],'gapfast':g[:,0],'gapslow':g[:,1],'stratum':st}
Q.update({f'p{j}':v for j,v in enumerate(p.T)}); Q.update({f'x{j}':v for j,v in enumerate(xx.T)})
Q.update({f'hch{j}':v for j,v in enumerate(np.asarray(di['hold_channel_raw']).T)})
Q.update({f'pch{j}':v for j,v in enumerate(np.asarray(di['past_channel_raw'])[:,-1,:].T)})
Q.update({f'str{j}':st==j for j in np.unique(st)})
Q.update({'hold*zslow':h*z[:,1],'hold*score':h*s,'zslow*score':z[:,1]*s,'hold2':h*h,'score2':s*s,'score3':s**3,'end-start':p[:,-1]-p[:,0]})
def evaluate(cols):
 A=np.stack([s[:,None]*a,s[:,None]*d]+[v[:,None]*b1 for v in cols],axis=-1)
 def fit(m): return np.linalg.lstsq(A[m].reshape(-1,A.shape[-1]),Y[m].ravel(),rcond=None)[0]
 beta=fit(one==1); e=Y-A@beta; eo=np.empty_like(e)
 for j in np.unique(l): eo[l==j]=Y[l==j]-A[l==j]@fit(l!=j)
 score=[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(eo[:,:-1]**2)),np.sqrt(np.mean(eo[:,-1]**2)),max(np.sqrt(np.mean(eo[l==j,:-1]**2)) for j in np.unique(l))]
 return np.array(score),beta,e
best=[]
for k,v in Q.items():
 sc,bta,_=evaluate([one,s,v]); best.append((sc[1],k,sc,bta))
for row in sorted(best,key=lambda x:x[0])[:16]:print(row)
for keys in [('hold','zslow'),('hold','gapslow'),('zfast','zslow'),('p0','p5'),('hold','stratum')]:
 print(keys,evaluate([one,s,*[Q[k] for k in keys]])[:2])
sc,bta,e=evaluate([one,s]); r=e@b1/(b1@b1); print('deviations (lab,stratum,score,hold,zslow,slow residual)',np.c_[l,st,s,h,z[:,1],r].round(3)); print('test score span',np.quantile(th[2]+x[n:]@th[3:],[0,.5,1]))
