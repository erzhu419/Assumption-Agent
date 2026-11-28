import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; t=s['eval_times']; X=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']];
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(h,y):return np.linalg.lstsq(h,y,rcond=None)[0]
a,b=s['panel_a']['u'],s['panel_a']['v']; g=np.array([np.median((b[1:,j]-b[0,j])/(a[1:,j]-a[0,j])) for j in range(4)]); o=np.median(b-a*g,axis=0); keep=np.linalg.norm(b-(a*g+o),axis=1)<.05; coef=fit(aff(a[keep]),b[keep]); print('A',keep.sum(),coef.round(5)); z=np.linalg.solve(coef[1:].T,(X-coef[0]).T).T
R=fit(aff(s['panel_b']['u']),s['panel_b']['v']); print('B svd',np.linalg.svd(R[1:],compute_uv=False));
variants={'sat':lambda x,k:np.tanh(k*x),'exp':lambda x,k:np.expm1(-k*x*x),'cos':lambda x,k:np.cos(k*x),'sin':lambda x,k:np.sin(k*x),'atan':lambda x,k:np.arctan(k*x),'rat':lambda x,k:x*x/(1+k*x*x)}
for key in ['v1','v2']:
 y=p[key]; print('panel',key)
 for label,f in variants.items():
  for lin in [False,True]:
   def design(k):return aff(np.c_[u,f(u,k)] if lin else f(u,k))
   def fun(k):
    h=design(k);return np.mean((h@fit(h,y)-y)**2)
   r=minimize_scalar(fun,bounds=(.01,4),method='bounded'); print(label,lin,round(r.x,5),round(r.fun**.5,7))
 for j in range(4):
  def pre(a):
   a0,w,shift,amp=a[0],a[1:5],a[5],a[6]; g=u@w+shift
   return a0+amp*(np.tanh(g) if key=='v1' else np.exp(-g*g))-y[:,j]
  w=fit(aff(u),y[:,j])[1:]; rng=np.random.default_rng(j)
  r=min([least_squares(pre,np.r_[0,v,0,amp],max_nfev=160) for v in [w,*rng.normal(size=(12,4))] for amp in [-1,1]],key=lambda r:np.linalg.norm(r.fun)); print('rotate_first',j,np.sqrt(np.mean(r.fun**2)),r.x.round(5))
q=-np.expm1(-t[:,None]*[.4125,1.80995]); A=np.c_[Y[:,0],fit(q,Y[:,1:].T).T];
from scipy.spatial.distance import cdist
for name,w in [('raw',X),('a',z),('b_forward',aff(z)@R),('b_inverse',(z-R[0])@np.linalg.inv(R[1:]))]:
 print('coordinates',name)
 for deg in [1,2,3,4]:
  f=aff(np.column_stack([np.prod(w[:,inds],axis=1) for d in range(1,deg+1) for inds in combinations_with_replacement(range(4),d)]));
  pred=f[52:]@fit(f[:52],A[:52]); print(deg,'amplitude MAE',np.mean(abs(pred-A[52:]),axis=0).round(4),'trajectory',np.mean(abs((pred[:,1:]-A[52:,1:])@q.T)).round(4))
 for l in [.35,.65,1.,1.5,2.5,4]:
  K=np.exp(-cdist(w,w,'sqeuclidean')/(2*l*l)); pred=K[52:,:52]@np.linalg.solve(K[:52,:52]+.0001*np.eye(52),A[:52]); print(l,np.mean(abs(pred-A[52:]),axis=0).round(4),np.mean(abs((pred[:,1:]-A[52:,1:])@q.T)).round(4))
