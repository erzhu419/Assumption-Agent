import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=130)
u=s['panel_a']['u'];v=s['panel_a']['v']; ij=np.argsort(u,axis=0); L=[]
for k in range(4):
 D=np.c_[np.ones(len(u)),u[:,k]]; w=least_squares(lambda w:D@w-v[:,k],[0,1],loss='cauchy',f_scale=.02).x; keep=abs(D@w-v[:,k])<.01; L.append(np.linalg.lstsq(D[keep],v[keep,k],rcond=None)[0]);print('align',k,w,keep.sum())
L=np.array(L);x=(s['train']['x']-L[:,0])/L[:,1];t=s['train']['traj'];Y=np.c_[s['train']['y'],t@np.linalg.svd(t,full_matrices=False)[2][:2].T];
def des(x,d,odd=False):return np.column_stack([np.ones(len(x))]+[x[:,j].prod(1) for k in range(1,d+1) if not odd or k%2 for j in combinations_with_replacement(range(4),k)])
for name,z,y in [('training',x,Y),('b',s['panel_b']['u'],s['panel_b']['v']),('c1',s['panel_c']['u'],s['panel_c']['v1'])]:
 print('\n',name)
 for d,odd in [(1,False),(2,False),(3,True),(3,False)]:
  A=des(z,d,odd);w=np.linalg.lstsq(A,y,rcond=None)[0]; E=y-A@w; H=np.sum(np.linalg.svd(A,full_matrices=False)[0]**2,1);print(d,odd,'mae',abs(E).mean(0),'loo',np.mean(abs(E)/np.maximum(1e-6,1-H[:,None]),0),'res_svd',np.linalg.svd(E,compute_uv=False));
  if d==3 and odd:print(list(zip(combinations_with_replacement(range(4),3),w[5:].round(3))))
  if d==1: print('coef',w,'respc',np.linalg.svd(E,full_matrices=False)[2]);print('odd/even under centered pair dist')
 def coord(z,f):return np.c_[np.ones(len(z)),f(z)]
 for typ in ('sin','tanh','cube','rational','sinh','atan'):
  def F(p):
   q=z*p[:4]+p[4:];return {'sin':np.sin,'tanh':np.tanh,'cube':lambda a:a+a**3,'rational':lambda a:a/(1+a*a),'sinh':np.sinh,'atan':np.arctan}[typ](q)
  def fun(p):
   a=coord(z,lambda _:F(p));return (a@np.linalg.lstsq(a,y,rcond=None)[0]-y).ravel()
  res=least_squares(fun,[.8]*4+[0]*4,bounds=([.03]*4+[-4]*4,[5]*4+[4]*4),max_nfev=180); print(typ,res.x,np.mean(abs(res.fun.reshape(y.shape)),0))
