import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; Y=np.c_[y,z]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,d):
 keys=[a for n in range(d+1) for a in it.combinations_with_replacement(range(x.shape[1]),n)]; return np.column_stack([np.prod(x[:,a],axis=1) for a in keys]),keys
def metrics(X):
 e=[]
 for te in [np.arange(52,72),*np.array_split(np.random.default_rng(42).permutation(72),5)]:
  tr=np.setdiff1d(np.arange(72),te); r=abs(X[te]@fit(X[tr],Y[tr])-Y[te]);e.append([r[:,0].mean(),r[:,1:].mean(),r[:,1:].mean(0).max()])
 return np.round(np.array([e[0],np.max(e[1:],axis=0)]),5)
p=s['panel_a']; A=poly(p['u'],1)[0]; b=fit(A,p['v']);
for _ in range(15):
 r=np.linalg.norm(A@b-p['v'],axis=1);w=1/np.sqrt(1+(r/max(2*np.median(r),1e-5))**2); b=fit(w[:,None]*A,w[:,None]*p['v'])
r=np.linalg.norm(A@b-p['v'],axis=1); k=np.argsort(r)[:-3];b=fit(A[k],p['v'][k]);print('A',b.round(4),'residual',np.sort(np.linalg.norm(A@b-p['v'],axis=1))[-6:].round(5)); u=np.linalg.solve(b[1:].T,(x-b[0]).T).T
p=s['panel_b']; bf=fit(poly(p['u'],1)[0],p['v']);print('B',bf.round(4));f=poly(x,1)[0]@bf
p=s['panel_c'];C,keys=poly(p['u'],2);v1=fit(C,p['v1']);v2=fit(C,p['v2']);
for j,a in enumerate(keys):print(a,v1[j].round(4),v2[j].round(4))
for inp in [x,u]:
 D=poly(inp,2)[0];g,h=D@v1,D@v2
 for tag,F in [('raw2',D),('raw3',poly(inp,3)[0]),('panel',np.c_[np.ones(len(x)),f,g,h]),('panel+pair',np.c_[np.ones(len(x)),f,g,h,(f[:,:,None]*g[:,None,:]).reshape(len(x),16)]),('panel+diag',np.c_[np.ones(len(x)),f,g,h,f*g,f*h]),('panel+v2pair',np.c_[np.ones(len(x)),f,g,h,(f[:,:,None]*h[:,None,:]).reshape(len(x),16)]),('panel+v1v2pair',np.c_[np.ones(len(x)),f,g,h,(f[:,:,None]*g[:,None,:]).reshape(len(x),16),(f[:,:,None]*h[:,None,:]).reshape(len(x),16)])]:print('inp', 'x' if inp is x else 'inverse-A',tag,metrics(F).tolist())
D,kk=poly(x,3);b=fit(D,Y);print('cubic coef y/t.25/t1.82/t3:');print(np.round(b[:,[0,1,5,8]],4));
for tag,func,p0 in [('two exp',lambda q: -np.expm1(-t[:,None]*np.sort(q)),[.35,.9]),('exp+lin',lambda q: np.c_[-np.expm1(-t*q[0]),t],[.5]),('exp+sin',lambda q: np.c_[-np.expm1(-t*q[0]),np.sin(t*q[1])],[.5,.8]),('exp+quad',lambda q: np.c_[-np.expm1(-t*q[0]),t*t],[.5]),('exp+exp derivative',lambda q: np.c_[-np.expm1(-t*q[0]),t*np.exp(-t*q[0])],[.5])]:
 def res(q):
  M=func(q);return (M@fit(M,z.T)-z.T).ravel()
 opt=least_squares(res,p0,bounds=(.01,4));print(tag,opt.x,np.sqrt(np.mean(opt.fun**2)))
