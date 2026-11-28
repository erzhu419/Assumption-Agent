exec(open('/tmp/probe.py').read().split('for g,ss in')[0])
from itertools import combinations
p,h,f=sd
r=f@th[2:]; tau0=tau.copy();
D=np.vstack((np.r_[1/tau,1],np.c_[-np.expm1(-t[:,None]/tau),t])); y=Y(di)-base(sd)
coef=np.linalg.lstsq(D,y.T,rcond=None)[0].T
k=np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0.][:,None]/tau)*(-np.expm1(-dt[:,None]/tau)); z=p@k; zr=p[:,::-1]@k
print('tau',tau,'kernel',k,'resid slopes on drift',np.linalg.lstsq(r[:,None],coef,rcond=None)[0]);
A=np.c_[h,p,f]; print('full effective coefficients [hold 6past 1 4feat]',np.linalg.lstsq(A,coef,rcond=None)[0]); print('full residual std',np.std(coef-A@np.linalg.lstsq(A,coef,rcond=None)[0],0))
features={'hold':h,'drift':r,'const':np.ones(len(h)),'diff':p[:,-1]-p[:,0],'avg':np.mean(p,axis=1),'dtavg':p@dt/dt.sum()}
for i in range(6): features['p'+str(i)]=p[:,i]
for i in range(2):
 features['z'+str(i)]=z[:,i]; features['zr'+str(i)]=zr[:,i]; features['g'+str(i)]=h-z[:,i];features['z-zr'+str(i)]=z[:,i]-zr[:,i]
for j in range(3):
 print('BRANCH',j)
 for depth in range(1,4):
  scores=[]
  for names in combinations(features,depth):
   X=np.column_stack([features[n] for n in names]); b=np.linalg.lstsq(X,coef[:,j],rcond=None)[0]; e=coef[:,j]-X@b
   scores.append((np.sqrt(np.mean(e**2)),names,b))
  for row in sorted(scores,key=lambda z:z[0])[:4]:print(row)
print('public asset note keys',list(__import__('json').load(open('/holdout_task/public_asset_index.json'))))
