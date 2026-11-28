import pickle,numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ii],axis=1) for j in range(1,d+1) for ii in combinations_with_replacement(range(x.shape[1]),j)])
a=s['panel_a']; u=aff(a['u']); v=a['v']; keep=np.ones(len(u),bool)
for _ in range(6):
 A=fit(u[keep],v[keep]); e=np.linalg.norm(u@A-v,axis=1); keep=e<np.median(e)*4
b=s['panel_b']; B=fit(aff(b['u']),b['v']); c=s['panel_c']; x=s['train']['x']; z=(x-A[0])@np.linalg.inv(A[1:]); h=(z-B[0])@np.linalg.inv(B[1:]); T=s['train']['traj']; y=s['train']['y']; U,D,V=svd(T,full_matrices=False); R=np.c_[y,U[:,:2]*D[:2]]
if __name__=='__main__':
 print('A',A,'residuals',np.sort(e)); print('B',B); print('times',s['eval_times']); print('V',V[:2]); print('x,z,h ranges',[(g.min(0),g.max(0),g.std(0)) for g in (x,z,h)]); print('c sample',np.c_[c['u'],c['v1'],c['v2']][:10]); print('corr c',np.corrcoef(np.c_[c['u'],c['v1'],c['v2']].T)[:4,4:]);
 for label,g in [('x',x),('z',z),('h',h),('x>B',x@B[1:]+B[0]),('z>B',z@B[1:]+B[0])]:
  for d in [1,2,3]:
   F=poly(g,d); q=fit(F[:52],R[:52]); print(label,d,'mae',np.mean(abs(F[52:]@q-R[52:]),0),'coef',q[:5].T)
 for label in ['v1','v2']:
  for d in [1,2]:
   F=poly(c['u'],d); q=fit(F,c[label]); print('C',label,d,'rmse',np.sqrt(np.mean((F@q-c[label])**2,axis=0)),q.T)
