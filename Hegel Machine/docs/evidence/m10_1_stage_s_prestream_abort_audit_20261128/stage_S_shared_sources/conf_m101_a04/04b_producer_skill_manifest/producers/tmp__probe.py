import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import svdvals
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=map(tr.get,['x','y','traj']); ts=s['eval_times']; np.set_printoptions(precision=5,suppress=True,linewidth=120)
def fit(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(X):return np.column_stack([np.ones(len(X)),X])
def poly(X,d):return np.column_stack([np.ones(len(X))]+[np.prod(X[:,j],axis=1) for i in range(1,d+1) for j in combinations_with_replacement(range(X.shape[1]),i)])
print('times',ts,'telemetry',s['telemetry_vec'],'traj sv',svdvals(T)); print('y, traj correlations',np.corrcoef(np.column_stack([y,T]).T)[0]); print('slices ranges',np.quantile(T,[0,.1,.5,.9,1],axis=0))
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; U=aff(p['u']);
 for j,V in p.items():
  if j=='u':continue
  sel=np.ones(len(U),bool)
  for _ in range(6):
   B=fit(U[sel],V[sel]); r=np.linalg.norm(U@B-V,axis=1); sel=r<=np.sort(r)[int(.8*len(U))-1]
  print(k,j,'B',B,'res quantile',np.quantile(r,[0,.25,.5,.75,.85,.9,1]),'corr',np.corrcoef(p['u'].T,V.T)[:4,4:]);print('quadratic 18/remaining',np.mean(abs(poly(p['u'],2)[18:]@fit(poly(p['u'],2)[:18],V[:18])-V[18:]),axis=0))
  if k=='panel_a': A=B
z=aff(x)@A
for k,Z in [('raw',x),('a',z),('a_inv', (x-A[0])@np.linalg.inv(A[1:]))]:
 for nm,W in [('lin',Z),('tanh',np.tanh(Z)),('sin',np.sin(Z))]:
  for d in (1,2,3):
   D=poly(W,d); P=D[56:]@fit(D[:56],np.c_[y,T][:56]); print(k,nm,d,'mae',np.mean(abs(P-np.c_[y,T][56:]),axis=0))
for i in range(8):
 b=fit(aff(x),T[:,i]);print('raw lin traj',i,b,np.std(T[:,i]-aff(x)@b))
