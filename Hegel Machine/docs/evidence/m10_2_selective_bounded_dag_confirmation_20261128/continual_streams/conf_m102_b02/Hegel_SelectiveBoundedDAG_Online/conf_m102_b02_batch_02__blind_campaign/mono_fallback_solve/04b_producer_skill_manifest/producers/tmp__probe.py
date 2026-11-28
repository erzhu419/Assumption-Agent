import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=135)
def poly(x,d=2):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,c],axis=1) for i in range(1,d+1) for c in combinations_with_replacement(range(x.shape[1]),i)])
def reg(u,v):
 c=np.linalg.lstsq(u,v,rcond=None)[0];r=v-u@c;return c,np.sqrt(np.mean(r*r,axis=0)),np.mean(abs(r/(1-np.sum(u*np.linalg.pinv(u).T,1)[:,None])),axis=0)
print('times tele',s['eval_times'],s['telemetry_vec']);u=s['panel_a']['u'];v=s['panel_a']['v']; D=poly(u,1);k=np.ones(len(u),bool)
for _ in range(5):
 a=np.linalg.lstsq(D[k],v[k],rcond=None)[0];e=np.linalg.norm(D@a-v,axis=1);k=e<3*np.median(e)
print('a',a,'res',np.sort(e));print('summary',s['train']['summary']);
u=s['panel_c']['u'];D=poly(u)
for key in ['v1','v2']:
 v=s['panel_c'][key];print(key,'range',v.min(0), v.max(0));c,r,l=reg(D,v);print('quadratic',c,'rmse',r,'loo',l)
 for name,f in [('tanh',np.tanh),('sin',np.sin),('erf',erf),('asinh',np.arcsinh),('linear',lambda x:x)]:
  for degree in [1,2]:
   M=poly(u,degree);cols=[]
   for j in range(4):
    y=v[:,j];p=np.r_[np.linalg.lstsq(M,y,rcond=None)[0],0,0];
    def pred(p):return M@p[:-2] if name=='linear' else p[-1]+np.exp(np.clip(p[-2],-20,20))*f(M@p[:-2]);
    q=least_squares(lambda p:pred(p)-y,p,max_nfev=500);cols.append(np.sqrt(np.mean(q.fun**2)))
   print(name,degree,np.array(cols))
X=s['train']['x'];T=s['train']['traj'];Y=s['train']['y']; t=s['eval_times'];
for key,fun in [('t, exp',lambda k:np.array([t,-np.expm1(-k*t)])),('1, exp',lambda k:np.array([np.ones(len(t)),-np.expm1(-k*t)])),('exp, exp',lambda k:np.array([-np.expm1(-k[0]*t),-np.expm1(-k[1]*t)]))]:
 def r(p):
  b=fun(p if len(p)>1 else p[0]);return (T-T@np.linalg.pinv(b)@b).ravel()
 q=least_squares(r,[2,5] if key=='exp, exp' else [2]); print(key,q.x,np.sqrt(np.mean(q.fun*q.fun)))
