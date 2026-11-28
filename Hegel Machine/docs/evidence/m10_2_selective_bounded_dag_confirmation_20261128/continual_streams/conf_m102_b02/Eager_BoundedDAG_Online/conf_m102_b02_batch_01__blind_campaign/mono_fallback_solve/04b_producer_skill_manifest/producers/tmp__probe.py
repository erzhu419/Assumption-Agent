import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=130)
def add(x):return np.c_[np.ones(len(x)),x]
A=s['panel_a']; U=add(A['u']); V=A['v']; ii=np.ones(len(U),bool)
for k in range(5):
 a=np.linalg.lstsq(U[ii],V[ii],rcond=None)[0];r=np.linalg.norm(U@a-V,axis=1);ii=r<3*np.median(r)
print('A',a,'errors',np.sort(r));print('times',s['eval_times']); print('train y',s['train']['y'][:20]);
for k in ['panel_b','panel_c']:
 p=s[k];U=add(p['u']);
 for j in p.keys()-{'u'}:
  Y=p[j]; print(k,j,'range',Y.min(0),Y.max(0),'first',Y[:4]);
  for name,f in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('asinh',np.arcsinh),('soft',lambda x:x/(1+abs(x))),('rational',lambda x:x/np.sqrt(1+x*x)),('square',lambda x:x*x)]:
   vals=[]
   for y in Y.T:
    b=np.linalg.lstsq(U,y,rcond=None)[0]
    def pred(q):return q[-2]*f(U@q[:5])+q[-1]
    fit=least_squares(lambda q:pred(q)-y,np.r_[b,1.,0.],max_nfev=400); vals.append(np.sqrt(np.mean(fit.fun**2)))
   print(name,np.array(vals))
