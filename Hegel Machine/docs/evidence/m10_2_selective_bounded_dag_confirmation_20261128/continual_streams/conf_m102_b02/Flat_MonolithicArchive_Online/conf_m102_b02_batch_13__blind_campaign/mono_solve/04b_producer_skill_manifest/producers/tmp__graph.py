import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']
def I(a):return np.c_[np.ones(len(a)),a]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def rms(a):return np.sqrt(np.mean(a*a))
def bfit(p,key,fn,q0,verbose=False):
 u=p['u']; v=p[key]
 def d(x,q):return I(np.c_[x,fn(q*x)])
 def residual(q):
  a=d(u,q);return (a@ls(a,v)-v).ravel()
 q=least_squares(residual,[q0],max_nfev=200,ftol=1e-12).x
 co=ls(d(u,q),v)
 if verbose:print(key,q,rms(residual(q)),np.round(co,4))
 return lambda x:d(x,q)@co
p=s['panel_a']; good=np.linalg.norm(p['v'],axis=1)<4
G=ls(I(p['u'][good]),p['v'][good]); z=(t['x']-G[0])@np.linalg.inv(G[1:]); B=bfit(s['panel_b'],'v',np.cos,2.6,True); C=bfit(s['panel_c'],'v1',lambda x:x/(1+abs(x)),.83,True); D=bfit(s['panel_c'],'v2',np.cos,1.28,True)
_,_,vt=np.linalg.svd(t['traj'][:52],full_matrices=False); Y=np.c_[t['y'],t['traj']@vt[:2].T]; Y[:,1:]/=np.linalg.norm(Y[:52,1:],axis=0)/np.linalg.norm(Y[:52,0]);Y=Y.copy()
w=B(z); groups={'z':z,'B':w,'Cz':C(z),'Dz':D(z),'CB':C(w),'DB':D(w),'Bz2':B(C(z)),'Bz3':B(D(z))};
for k,a in groups.items():
 for p in (1,2,3):
  X=I(np.column_stack([np.prod(a[:,ij],axis=1) for j in range(1,p+1) for ij in combinations_with_replacement(range(4),j)])); co=ls(X[:52],Y[:52]); print(k,p,'rms',np.round(np.std(X[:52]@co-Y[:52],axis=0),4),'MAE',np.round(np.mean(abs(X[52:]@co-Y[52:]),axis=0),4))
for key in ['z','B','Cz','Dz','CB','DB']:
 for key2 in ['z','B','Cz','Dz','CB','DB']:
  if key>=key2:continue
  a,b=groups[key],groups[key2]; X=I(np.c_[a,b,(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)]); co=ls(X[:52],Y[:52]); print(key,key2,np.round(np.std(X[:52]@co-Y[:52],axis=0),4),np.round(np.mean(abs(X[52:]@co-Y[52:]),axis=0),4))
np.savez('/tmp/graph.npz',**groups,Y=Y,vt=vt[:2])
