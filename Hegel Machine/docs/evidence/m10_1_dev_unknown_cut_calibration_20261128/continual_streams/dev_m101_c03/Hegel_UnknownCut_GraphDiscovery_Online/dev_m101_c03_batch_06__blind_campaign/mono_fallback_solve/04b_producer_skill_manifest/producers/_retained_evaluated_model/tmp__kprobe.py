from graphprobe import *
from itertools import combinations_with_replacement as cr
from scipy.spatial.distance import cdist
T=S['train']; t=S['eval_times']; h=T['traj']; y=T['y']; Y=np.c_[y,h]; ai,af,B,C,D=ops(S); z=ai(T['x']); b=B(z); c=C(z); d=D(z); n=len(z); cd=np.c_[c,d]; dot=(c*d).sum(1)
def p(x,k):return np.stack([np.prod(x[:,i],axis=1) for i in cr(range(x.shape[1]),k)],axis=1)
v=-np.expm1(-t[:,None]*np.array([.396,1.7987,5.27])); A=fit(v,h.T).T
print('y~amps',fit(one(A),y),np.std(y-one(A)@fit(one(A),y))); print('summary',T['summary']); print('telemetry',S['telemetry_vec']); print('sv',np.linalg.svd(h,compute_uv=False))
F={'lin':z,'quad':p(z,2),'cubic':p(z,3),'cd':cd,'dot':dot[:,None],'CD2':p(cd,2),'cz':(c[:,:,None]*z[:,None,:]).reshape(n,-1),'dz':(d[:,:,None]*z[:,None,:]).reshape(n,-1),'bcd':np.c_[b*c,b*d],'rat':np.c_[c/(1+d*d),d/(1+c*c)],'pow4':p(z,4),'pow5':p(z,5)}
for name,X in F.items():
 X=one(X); R=np.c_[y,A]-X@fit(X,np.c_[y,A]); print(name,'RMS',R.std(0))
base=np.c_[F['quad'],F['cubic'],cd,dot]; m=one(base); ry=y-m@fit(m,y); rh=h-m@fit(m,h); print('base residual y/h',np.std(ry),rh.std(0),np.linalg.svd(rh,compute_uv=False)); print('resid corr',np.corrcoef(np.c_[ry,rh].T)[0]);
folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),ix),ix) for ix in np.array_split(np.random.default_rng(123).permutation(72),5)];
def kernel(X):
 X=X-X.mean(0); return (X@X.T)/np.mean(np.sum(X*X,axis=1))
K={k:kernel(X) for k,X in F.items()};
for key,X in [('z',z),('cd',cd),('all',np.c_[z,cd])]:
 Q=cdist(X,X,'sqeuclidean');
 for f in [.3,1,3,9]:K[f'rbf{key}{f}']=np.exp(-f*Q/np.median(Q))
K0=K['quad']+K['cubic']+K['cd']+K['dot']; rows=[]
def evalk(G,l):
 E=[]
 for tr,te in folds:
  A=G[np.ix_(tr,tr)]+l*np.eye(len(tr)); Q=G[np.ix_(te,tr)]; coef= np.linalg.solve(A,np.c_[Y[tr],np.ones(len(tr))]); mu=coef[:,:-1].sum(0)/coef[:,-1].sum(); P=mu+Q@(coef[:,:-1]-coef[:,-1,None]*mu); e=abs(P-Y[te]); E.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
 E=np.array(E); return E[0],E.mean(0),E.max(0)
for name, G in {'base':K0,'plain':K['quad']+K['cd']+K['dot'],**{k:V for k,V in K.items() if k.startswith('rbf')}}.items():
 for l in [.001,.01,.06,.25]:
  e,m,w=evalk(G,l); rows.append((sum(m[:2]),name,0,l,e,m,w))
for nm,L in {**K,'mix':K['cz']+K['dz']+K['CD2'],'odd':K['lin']+K['dz']+K['cz'],'even':K['CD2']+K['pow4'],'allextra':K['CD2']+K['pow4']+K['cz']+K['dz']+K['rat']}.items():
 for w in [.003,.02,.1,.5,2]:
  for l in [.001,.015,.08]:
   e,m,ww=evalk(K0+w*L,l); rows.append((sum(m[:2]),nm,w,l,e,m,ww))
print('Top joint');
for row in sorted(rows)[:20]:print(row)
print('Top y');
for row in sorted(rows,key=lambda x:x[-2][0])[:12]:print(row)
print('Top h');
for row in sorted(rows,key=lambda x:x[-2][1])[:12]:print(row)
