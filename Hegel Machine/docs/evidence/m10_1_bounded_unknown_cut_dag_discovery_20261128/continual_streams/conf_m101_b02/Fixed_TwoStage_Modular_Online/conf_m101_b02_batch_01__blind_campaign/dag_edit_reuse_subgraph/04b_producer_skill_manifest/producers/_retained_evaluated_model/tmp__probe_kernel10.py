import pickle,runpy,numpy as np
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path('/holdout_task/node_00.py'); y=s['train']['y']; traj=s['train']['traj']; n=len(y)
o,A=old['_calibrate'](s['panel_a']); z=np.linalg.solve(A.T,(s['train']['x']-o).T).T; b,c=s['panel_b'],s['panel_c']; pf=old['_panel_fit']
B=[lambda x:pf(b['u'],b['v'],x,fn) for fn in ()]
def features(mode):
 bt=[lambda u:(np.sin(2.815*u),),lambda u:(np.tanh(3.51*u),u/(1+(2.434*u)**2)),lambda u:(np.sin(2.815*u),u/(1+(2.434*u)**2))][mode]
 f=lambda x:pf(b['u'],b['v'],x,bt); p=lambda x:pf(c['u'],c['v1'],x,lambda u:(np.sin(1.5*u),u*abs(u))); q=lambda x:pf(c['u'],c['v2'],x,lambda u:(u*u,np.cos(2.5*u)))
 F=f(z); C=p(z); D=q(z); H=p(F); J=q(F); return [z,F,C,D,H,J,p(.5*(F+z)),q(.5*(F+z)),q(H)]
def norm(a):return (a-a.mean(0))/np.maximum(a.std(0),1e-7)
def cov(a):return a@a.T
folds=[*np.array_split(np.random.default_rng(38).permutation(n),5),*np.array_split(np.random.default_rng(72).permutation(n),5),np.arange(n-20,n)]
fold=[]
for v in folds:
 t=np.setdiff1d(np.arange(n),v); basis=np.linalg.svd(traj[t],full_matrices=False)[2][:2]; T=np.c_[y[t],traj[t]@basis.T]; mu=T.mean(0); fold.append((t,v,basis,T-mu,mu))
def predict(k,t,v,T,mu,lam):
 a=k[np.ix_(t,t)]; b=k[np.ix_(v,t)]; m=a.mean(0); a=a-m[:,None]-m+m.mean(); b=b-b.mean(1)[:,None]-m+m.mean(); a.flat[::len(t)+1]+=lam;return mu+b@np.linalg.solve(a,T)
def score(k,f,c):
 E=[]
 for t,v,basis,T,mu in fold:
  P=.5*predict(k,t,v,T,mu,.0003)+(predict(k+f,t,v,T,mu,.003)+predict(k+c,t,v,T,mu,.0003)+predict(k+f+c,t,v,T,mu,.001))/6
  E.append([np.mean(abs(y[v]-P[:,0])),np.mean(abs(traj[v]-P[:,1:]@basis))]);
 return np.array(E)
results=[]
for mode in range(3):
 Q=features(mode); N=list(map(norm,Q)); KS=list(map(cov,Q)); base=sum(KS[:5]); f=KS[1];c=KS[2]+KS[3]
 for ids in [(1,2,3,4),(0,1,2,3,4),(1,4),(1,3,4),(1,4,5),(1,2,3,4,5),(1,4,6,7),(0,1,4,5,8),(4,),(4,5)]:
  X=np.column_stack([N[i] for i in ids]); d=cdist(X,X,'sqeuclidean')/X.shape[1]; dots=cov(X)/X.shape[1]; diff=(X[:,None,:]-X[None,:,:])**2
  for w in (.045,.09,.18,.3,.55,1.):
   for kname,k in [('rbf',np.exp(-w*d)),('add',np.exp(-w*diff).mean(2)),('blend',.55*np.exp(-w*d)+.45*np.exp(-w*diff).mean(2)),('poly',(1+w*dots)**3)]:
    for amp in (.3,1.,3.):
     E=score(base+amp*k,f,c); v=E[:-1]; loss=np.mean(v)+.30*np.max(v); results.append((loss,mode,ids,w,kname,amp,E[-1],np.mean(v,axis=0),np.max(v,axis=0)))
 print('mode',mode,flush=True)
for r in sorted(results,key=lambda v:v[0])[:16]:print(r)
print('public probe only:');
for r in sorted(results,key=lambda v:max(v[-3]))[:8]:print(r)
