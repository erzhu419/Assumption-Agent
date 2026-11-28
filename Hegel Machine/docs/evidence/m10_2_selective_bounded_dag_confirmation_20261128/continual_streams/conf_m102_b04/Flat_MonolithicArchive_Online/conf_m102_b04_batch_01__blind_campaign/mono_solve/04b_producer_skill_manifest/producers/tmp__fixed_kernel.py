from finalcv import *
from itertools import product
c=m._coordinates(s,x); g=[f@f.T for f in m._panel_features(s,c)]; D=[m._sqdist(z) for z in c]
B=[]
for i,j in sp:
 p=m.solve_batch(dict(s,train={**tr,**{k:tr[k][i] for k in ['x','y','traj']}},test={'x':x[j]})); B.append(np.c_[p['y_hat'],p['state_pred']])
B=np.array(B); truth=np.array([Y[j] for i,j in sp]);
def metrics(p):
 e=abs(p-truth); return np.c_[e[:,:,0].mean(1),e[:,:,1:].mean((1,2)),e[:,:,-1].mean(1)]
def pred(K,pen,temporal):
 P=[]
 for i,j in sp:
  t=Y[i]; mu=t.mean(0); k=K[np.ix_(i,i)]+pen*np.eye(len(i)); h=K[np.ix_(j,i)]; p=mu+h@np.linalg.solve(k,t-mu)
  if temporal:
   _,_,V=np.linalg.svd(t[:,1:]-mu[1:],full_matrices=False); q=V[0]; u=np.c_[np.ones(len(x)),c[1],m._panel_features(s,c)[3][:,:4]]; R=t[:,1:]-mu[1:]; R=R-np.outer(R@q,q); p[:,1:]=mu[1:]+np.outer((p[:,1:]-mu[1:])@q,q)+u[j]@np.linalg.solve(u[i].T@u[i]+.4*np.eye(u.shape[1]),u[i].T@R)
  P.append(p)
 return np.array(P)
results=[]
for lam,cal,h,pen,w,temporal in product([.5,.8,1.],[0.,.5,1.],[.12,.20,.30],[.00003,.0003,.001],[.15,1.2],[False,True]):
 d=(1-cal)*((1-lam)*D[0]+lam*D[2])+cal*((1-lam)*D[1]+lam*D[3]); f=(1-cal)*(.3*g[0]+.7*g[2])+cal*(.3*g[1]+.7*g[3]); P=pred(np.exp(-h*d)+w*f,pen,temporal); E=metrics(.55*B+.45*P); results.append((E,(lam,cal,h,pen,w,temporal)))
for j in [0,1,2]:
 print('TARGET',j)
 for e,p in sorted(results,key=lambda z:z[0][1:,j].mean()+.45*z[0][1:,j].max())[:6]:print(p,e[0],e[1:].mean(0),e[1:].max(0))
print('BASE',metrics(B)[0],metrics(B)[1:].mean(0),metrics(B)[1:].max(0))
# Direct scientific ablation at otherwise fixed kernel settings.
for w in [0,.25,2]:
 p=pred(np.exp(-D[3]/4)+w*g[3],.0003,False); e=metrics(p); print('v2 ablation',w,e[0],e[1:].mean(0),e[1:].max(0))
