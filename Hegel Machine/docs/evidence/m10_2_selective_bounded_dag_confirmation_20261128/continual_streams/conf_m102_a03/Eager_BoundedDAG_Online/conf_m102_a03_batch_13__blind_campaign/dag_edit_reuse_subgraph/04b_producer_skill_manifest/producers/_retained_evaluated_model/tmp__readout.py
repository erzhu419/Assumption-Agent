import pickle,numpy as np,sys,itertools
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00 as f
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; A=f._affine(u); x=f._inverse(s['train']['x'],f._fit_sensor(s['panel_a'])); y=s['train']['y']; tr=s['train']['traj']; U,D,V=np.linalg.svd(tr,full_matrices=False); H=np.c_[y,U[:,:2]*D[:2]]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,k):
 return np.column_stack([np.prod(x[:,j],axis=1) for n in range(k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),n)])
def cv(A,H,ids=None):
 if ids is None:ids=np.arange(len(A))
 i=ids[:-20];j=ids[-20:];return np.round(abs(A[j]@fit(A[i],H[i])-H[j]).mean(0),5)
def aff(x):return np.c_[np.ones(len(x)),x]
def bbase(q):return np.c_[aff(q),np.cos(2.6*q)]
def cbase(q):return np.c_[aff(q),np.tanh(2.15*q)]
def dbase(q):return np.c_[aff(q),np.cos(1.25*q)]
p=s['panel_b'];pc=s['panel_c'];z=bbase(x)@fit(bbase(p['u']),p['v']); c=cbase(x)@fit(cbase(pc['u']),pc['v1']); d=dbase(x)@fit(dbase(pc['u']),pc['v2']);
rng=np.random.default_rng(42); splits=np.array_split(rng.permutation(len(x)),5)
def stats(a,t):
 scores=[]
 for j in splits:
  i=np.setdiff1d(np.arange(len(x)),j);scores.append(np.mean(abs(a[j]@fit(a[i],t[i])-t[j])))
 return np.round([np.mean(scores),max(scores),np.mean(abs(a@fit(a,t)-t))],5)
for k,q in [('b',z),('c',c),('d',d),('raw',x),('b+d',np.c_[z,d]),('b+c',np.c_[z,c])]:
 for deg in range(1,4 if q.shape[1]==4 else 3):
  a=poly(q,deg); print(k,deg,stats(a,y),stats(a,tr))
 if q.shape[1]!=4:continue
 for name,r in [('bilin',q[:,0]*q[:,2]+q[:,1]*q[:,3]),('bilindiff',q[:,0]*q[:,2]-q[:,1]*q[:,3]),('neighbor',q[:,0]*q[:,1]+q[:,2]*q[:,3]),('outer',q[:,0]*q[:,3]+q[:,2]*q[:,1])]:
  print(k,name,stats(np.c_[aff(q),r],y),stats(np.c_[aff(q),r],tr))
 print('quadratic scalar coefficients',np.round(fit(poly(q,2),y),4));print('linear traj coeff first last',np.round(fit(aff(q),tr)[:,[0,-1]],4))
A=bbase(x); print('bbase scalar',stats(A,y)); print('bbase coeff scalar',fit(A,y));print('bbase traj',stats(A,tr));print('bbase traj coeff',np.round(fit(A,tr),4))
