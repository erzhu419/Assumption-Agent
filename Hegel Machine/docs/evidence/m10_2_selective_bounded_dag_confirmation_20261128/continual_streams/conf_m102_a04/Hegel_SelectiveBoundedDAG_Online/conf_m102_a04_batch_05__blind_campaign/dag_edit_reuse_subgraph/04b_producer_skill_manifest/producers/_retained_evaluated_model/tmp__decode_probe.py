import itertools as it, numpy as np, pickle, runpy, glob
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=np.load('/tmp/latent.npy'); old=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); p=s['panel_c'];u=p['u']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y);N=len(z)
def lin(x,y):return np.linalg.lstsq(x,y,rcond=1e-9)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def cy(x):return x*x[:,[1,2,3,0]]
def v2(x,r):return one(np.cos(r*x))
r=least_squares(lambda q:(v2(u,q)@lin(v2(u,q),p['v2'])-p['v2']).ravel(),[2.4]).x
c=one(cy(z))@lin(one(cy(u)),p['v1']); d=v2(z,r)@lin(v2(u,r),p['v2']); pb=s['panel_b']; b=one(z)@lin(one(pb['u']),pb['v']);
B=old['_time_basis'](s['eval_times'],Y[:,1:]); tar=np.c_[Y[:,0],lin(B,Y[:,1:].T).T];print('target corr',np.round(np.corrcoef(tar.T),3));
flds=[(np.arange(52),np.arange(52,72))]+[(np.delete(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(1).permutation(n),4)]
def pol(w,k):return np.array([w[:,c].prod(1) for a in range(1,k+1) for c in it.combinations_with_replacement(range(w.shape[1]),a)]).T
def score(name,F,kernel=False):
 F=(F-F[:n].mean(0))/np.maximum(.025,F[:n].std(0)); G=1+F@F.T
 if kernel:
  G=G/(F.shape[1]+1); y=G.diagonal(); dist=np.maximum(y[:,None]+y-2*G,0); G=(1+G*.25)**2+np.exp(-.06*dist)
 err=[]; best=None
 for reg in [1e-7,1e-4,.003,.03,.3,3,30]:
  E=[]
  for i,j in flds:
   fit=G[np.ix_(j,i)]@np.linalg.solve(G[np.ix_(i,i)]+reg*np.eye(len(i)),tar[i]); E.append(np.mean(abs(np.c_[fit[:,0],fit[:,1:]@B.T]-Y[j]),axis=0))
  E=np.array(E); v=np.c_[E[:,0], E[:,1:].mean(1)]; key=v.max();
  if best is None or key<best[0]:best=(key,reg,v)
 print(name,F.shape[1],np.round(best[2],4).tolist(),best[1])
for name,w in [('z',z),('c',c),('d',d),('cd',np.c_[c,d]),('zcd',np.c_[z,c,d]),('bcd',np.c_[b,c,d]),('primitive',np.c_[cy(z),np.cos(2.4*z)])]:
 for k in (1,2,3):score(name+str(k),pol(w,k))
 score(name+'kernel',w,True)
for name,w in [('concat',np.c_[z,c,d]),('cd',np.c_[c,d]),('bcd',np.c_[b,c,d])]:
 for fun in [np.sin,np.tanh,np.cos]:
  for f in [.5,1,2]:score(name+fun.__name__+str(f),np.c_[w,fun(f*w),pol(fun(f*w),2)])
F=one(pol(np.c_[c,d],2)); co=lin(F[:n],tar); print('cd quad in sample',np.sqrt(np.mean((F[:n]@co-tar)**2,0))); nn=[()]+[a for i in (1,2) for a in it.combinations_with_replacement(range(8),i)];
for k in range(3):print('top',k,[(nn[j],round(co[j,k],4)) for j in np.argsort(abs(co[:,k]))[-14:][::-1]])
print('time basis',B[:3])
