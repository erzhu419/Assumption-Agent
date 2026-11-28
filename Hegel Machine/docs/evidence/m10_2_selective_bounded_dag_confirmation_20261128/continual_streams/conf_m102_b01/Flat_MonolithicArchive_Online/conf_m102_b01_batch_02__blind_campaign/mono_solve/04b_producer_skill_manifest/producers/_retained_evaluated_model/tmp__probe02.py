import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; np.set_printoptions(suppress=True,precision=5,linewidth=150)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d=2):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ii],1) for k in range(1,d+1) for ii in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def cal(a,x):
 W=[]
 for u,v in zip(a['u'].T,a['v'].T):
  A=aff(u); q=ls(A,v)
  for _ in range(8):
   i=np.argsort(abs(A@q-v))[:42]; q=ls(A[i],v[i])
  keep=abs(A@q-v)<.002;q=ls(A[keep],v[keep]);W.append(q)
 return (x-np.array(W)[:,0])/np.array(W)[:,1]
z=cal(s['panel_a'],tr['x']); b=s['panel_b']; w=aff(z)@ls(aff(b['u']),b['v']); c=s['panel_c']; u=c['u']; q=ls(poly(u),c['v1']); r=ls(poly(u),c['v2']); print('v1 coefficients',q,'v2 coeff',r,sep='\n'); Y=np.c_[tr['y'],tr['traj']]
for name,x in [('raw',tr['x']),('z',z),('w',w)]:
 for op,p in [('v1',q),('v2',r)]:
  h=poly(x)@p
  for kind,F in [('aff',aff(h)),('plus_x',np.c_[aff(h),x]),('quad',poly(h)),('quad+rawquad',np.c_[poly(h),poly(x)[:,1:]]),('pair02_13',np.c_[aff(h),h[:,0]*h[:,2],h[:,1]*h[:,3]]),('inv',np.c_[aff(h),np.arctan(h),np.tanh(h)])]:
   B=ls(F[:52],Y[:52]); R=Y[52:]-F[52:]@B; print(name,op,kind,'ho',np.mean(abs(R[:,0])),np.mean(abs(R[:,1:])),'in',np.mean(abs((Y[:52]-F[:52]@B)[:,0])))
print('full quadratic coefficients',ls(poly(z),Y),sep='\n')
for k in [1,2,3]:
 V=np.linalg.svd(Y[:,1:],full_matrices=False)[2][:k]; print('rank',k,'reconstruction mae',np.mean(abs(Y[:,1:]-(Y[:,1:]@V.T)@V)))
