import pickle,numpy as np
from itertools import combinations as C, combinations_with_replacement as CR
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=140)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a']; u,v=p['u'],p['v']; L=[]
for j in range(4):
 a=np.c_[u[:,j]*0+1,u[:,j]]; w=least_squares(lambda w:a@w-v[:,j],[0,1],loss='cauchy',f_scale=.02).x; k=abs(a@w-v[:,j])<.01; L.append(np.linalg.lstsq(a[k],v[k,j],rcond=None)[0]);
b,g=np.array(L).T
tr=s['train']; x=(tr['x']-b)/g; Y=np.c_[tr['y'],tr['traj']];
def q(x,js):return np.column_stack([x[:,j].prod(1) for j in js])
def one(x):return np.ones((len(x),1))
pairs=list(C(range(4),2)); triples=list(C(range(4),3));
features={'diag cubic':lambda z: np.c_[one(z),z,z**3], 'diag cubic center':lambda z: np.c_[one(z),z,z**2,z**3], '+pairs':lambda z:np.c_[one(z),z,z**2,z**3,q(z,pairs)], '+triples':lambda z:np.c_[one(z),z,z**2,z**3,q(z,pairs),q(z,triples)], '+diag5':lambda z:np.c_[one(z),z,z**2,z**3,z**4,z**5,q(z,pairs),q(z,triples)], 'warped0.22':lambda z:np.c_[one(z),.22*z+z**3], 'warped.22+quad':lambda z:np.c_[one(z),.22*z+z**3,q(.22*z+z**3,list(CR(range(4),2)))]}
for pk,names in [('panel_b',['v']),('panel_c',['v1','v2'])]:
 z=s[pk]['u']
 for name in names:
  y=s[pk][name]; print('\n',pk,name)
  for key,f in features.items():
   A=f(z); w=np.linalg.lstsq(A,y,rcond=None)[0]; res=A@w-y; h=np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)]; print(key,np.mean(abs(res)),np.mean(abs(res)/np.maximum(1e-10,1-np.sum(h*h,axis=1)[:,None])))
   if key=='diag cubic':print('weights',w,'ratio',np.sum(w[1:5]*w[5:9],1)/np.sum(w[5:9]**2,1));
print('\ntraining')
folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(17).permutation(72),6));
for key,f in features.items():
 A=f(x); w=np.linalg.lstsq(A,Y,rcond=None)[0]; print(key, np.mean(abs(A@w-Y),axis=0)[[0,1,-1]]); es=[]
 for ti in folds:
  ri=np.setdiff1d(np.arange(72),ti); e=abs(A[ti]@np.linalg.lstsq(A[ri],Y[ri],rcond=None)[0]-Y[ti]); es.append([e[:,0].mean(),e[:,1:].mean()]);
 print('official / worst',es[0],np.max(es,axis=0));
 if key in ('diag cubic','+pairs'):print('weights y,t1,t8',w[:,[0,1,-1]]);
