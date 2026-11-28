import numpy as np,pickle,runpy
from scipy.optimize import minimize_scalar
m=runpy.run_path('/holdout_task/batch_solver.py');s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];q=m['_coordinates'](s,np.r_[tr['x'],s['test']['x']])[:72];d=m['_quadratic'](q);y=tr['y'];T=tr['traj'];LS=m['_fit'];t=s['eval_times'];ix=np.arange(len(y));rng=np.random.default_rng(574)
splits=[np.arange(52,72)]+[rng.choice(ix,20,False) for _ in range(20)]+[np.argsort(a)[-k:] for a in (np.linalg.norm(q,axis=1),np.max(abs(tr['x']),axis=1),*abs(q.T)) for k in (8,12)]
pair=d[:,7]+d[:,11]; base=np.c_[d[:,:5],pair]
for name,a in [('tied pairs',base),('separate pairs',d[:,[0,1,2,3,4,7,11]]),('tied + interactions',np.c_[base,d[:,[8,14]]]),('tied + cubic',np.c_[base,q**3])]:
 e=[]
 for te in splits:
  ti=np.setdiff1d(ix,te);e.append(np.mean(abs(a[te]@LS(a[ti],y[ti])-y[te])))
 print(name,e[0],max(e[1:21]),max(e[21:]),'full coeff',LS(a,y))
for kind in ('tanh','sin','softsign','arctan'):
 fn={'tanh':np.tanh,'sin':np.sin,'softsign':lambda u:u/(1+u*u),'arctan':np.arctan}[kind];e=[];weights=[]
 for te in splits:
  ti=np.setdiff1d(ix,te);a=base[ti];yy=y[ti]; l=LS(a,yy)
  from scipy.optimize import least_squares
  def pred(a,p):
   v=a[:,1:5]@p[1:5]; z=p[6];return p[0]+(fn(v*z)/z if z else v)+a[:,-1]*p[5]
  p=least_squares(lambda p:pred(a,p)-yy,np.r_[l,.2],bounds=(np.r_[[-np.inf]*6,0],np.r_[[np.inf]*6,3])).x
  e.append(np.mean(abs(pred(base[te],p)-y[te])));weights.append(p[-1])
 print(kind,e[0],max(e[1:21]),max(e[21:]),'nonlinearity',min(weights),max(weights))
D=m['_aff'](q); fit=D@LS(D,T);print('trajectory in-span sv',np.linalg.svd(fit,compute_uv=False));
def exp(a):return -np.expm1(-a*t)
for a in (2.15,1.25,.22):
 for name,b in [('t',t),('t*exp(1.25)',t*np.exp(-1.25*t)),('exp(.22)',exp(.22)),('constant',np.ones_like(t))]:
  A=np.c_[exp(a),b];print(a,name,np.sqrt(np.mean((A@LS(A,fit.T)-fit.T)**2)))
