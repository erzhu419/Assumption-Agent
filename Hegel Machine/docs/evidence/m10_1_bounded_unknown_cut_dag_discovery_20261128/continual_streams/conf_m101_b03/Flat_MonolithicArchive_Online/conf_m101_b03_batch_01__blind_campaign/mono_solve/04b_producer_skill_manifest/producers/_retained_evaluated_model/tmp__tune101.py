import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); import batch_solver as m
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; h,amp=m._modes(s['eval_times'],tr); Y=np.c_[y,amp]; z,c,d=m._coordinates(s,x)
sp=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=123).split(x)]
base=np.c_[m._quadratic(z),m._quadratic(c),m._quadratic(d)]
pairs=[(i,j) for i,j in combinations(range(4),2)]
F={'base':base,'cross':np.c_[base,m._quadratic(np.c_[z,c,d])[:,12:]],'diag':np.c_[base,z*c,z*d,c*d],'sines':np.c_[base,*[fn(f*z) for fn in (np.sin,np.cos) for f in (1.5,2.5)]],'cube':np.c_[base,z**3,c**3,d**3],'cross_d':np.c_[base,(c[:,:,None]*d[:,None,:]).reshape(len(z),-1)],'rat':np.c_[base,*[q/(1+k*q**2) for q in (z,c,d) for k in (.5,2)],*[1/(1+k*q**2) for q in (z,c,d) for k in (.5,2)]]}
for nm,f in F.items():
 for a in [.00045,.0015,.003,.006,.012]:
  E=[]
  for i,j in sp:
   p=m._readout(f[np.r_[i,j]],Y[i],len(i),a); e=abs(p[:,0]-y[j]); g=abs(p[:,1:]@h.T-tr[j]); E.append([e.mean(),g.mean()])
  E=np.array(E);print(nm,a,E[:2].round(4).tolist(),E[2:].mean(0).round(4),E[2:].max(0).round(4))
print('base coefficients')
labels=[f'{p}{i}' for p in 'zcd' for i in range(4)]
labels=[a for k in range(0,12,4) for a in labels[k:k+4]+[labels[i]+labels[j] for i in range(k,k+4) for j in range(i,k+4)]]
for target in Y.T:
 sc=base.std(0); model=ElasticNet(alpha=.003,l1_ratio=.8,max_iter=15000,tol=1e-6).fit((base-base.mean(0))/sc,target/target.std()); co=model.coef_; print([(labels[k],round(co[k],3)) for k in np.argsort(-abs(co))[:18]])
