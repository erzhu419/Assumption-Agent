exec(open('/tmp/periodic_probe.py').read().split("for a in ('x','y'):")[0])
rhs=src.transpose(0,2,1).reshape(len(src),-1).T
obs=cal(s['y_obs_raw']); Z=s['Z_nuisance']; cap=s['u_box']; total=s['total_budget']; nobs=len(obs)
from scipy.optimize import minimize
fold=(np.arange(nobs)//4)%4
models={'nominal':([],[]),'one_robin':([[0,3,6,9]],[]),'axis_robin':([[0,3],[6,9]],[]),'face_robin':([[0],[3],[6],[9]],[]),'axis_robin_face_exchange':([[0,3],[6,9],[1],[4],[7],[10]],[]),'face_robin_and_wrap':([[0],[3],[6],[9],[2,5],[8,11]],[4,5]),'face_robin_and_exchange':([[0],[3],[6],[9],[1],[4],[7],[10]],[])}
def fit(train,groups,wrap):
 z=Z[train]; p=np.linalg.pinv(z); b=obs[train]-z@(p@obs[train]);
 def state(x):
  a=x0.copy()
  for inds,v in zip(groups,x):a[inds]=v
  A=pred(a); d=A[train]-z@(p@A[train]); w=m._weights(d,b,1e-8,cap,total); return d@w-b, A@w+Z@(p@(obs[train]-A[train]@w))
 x=np.array([x0[g[0]] for g in groups]);
 if len(x): x=least_squares(lambda v:state(v)[0],x,bounds=(np.zeros(len(x)),np.where(np.isin(np.arange(len(x)),wrap),4.,8.)),diff_step=1e-4,max_nfev=40,gtol=1e-9).x
 return state(x)[1]-obs,x
for name,(g,w) in models.items():
 e,a=fit(np.ones(nobs,bool),g,w); cv=np.empty_like(e)
 for j in range(4):
  test=fold==j; cv[test]=fit(~test,g,w)[0][test]
 print(name,'coeff',a.round(4),'in-sample',np.sqrt(np.mean(e**2)),'heldout shift rms',np.sqrt(np.mean(cv.reshape(3,-1)**2,axis=1)))
