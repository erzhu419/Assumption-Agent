import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr=s['train']; X=tr['x']; Y=np.c_[tr['y'],tr['traj']]; t=s['eval_times'];a=s['panel_a'];u,v=a['u'],a['v']; keep=np.max(abs(v-(np.median(v-u,0)+u)),1)<1; b,g=np.array([np.linalg.lstsq(np.c_[np.ones(sum(keep)),u[keep,j]],v[keep,j],rcond=None)[0] for j in range(4)]).T
z=(X-b)/g
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def aff(x): return np.c_[np.ones(len(x)),x]
def loss(A,y):
 p=np.linalg.pinv(A); B=p@y; e=y-A@B; h=(A*p.T).sum(1);return np.linalg.norm(e)/np.sqrt(e.size),np.linalg.norm(e/np.maximum(1-h[:,None],1e-10))/np.sqrt(e.size)
fs={'exp':lambda u,k:np.expm1(k*u)/k, 'tanh':lambda u,k:np.tanh(k*u)/k,'sin':lambda u,k:np.sin(k*u)/k,'sinh':lambda u,k:np.sinh(k*u)/k,'rat':lambda u,k:u/(1+k*u*u),'cubic':lambda u,k:u+k*u**3,'quad':lambda u,k:u+k*u*u,'powexp':lambda u,k:u*np.exp(k*u),'softplus':lambda u,k:(np.logaddexp(0,k*u)-np.log(2))*2/k}
outputs={}
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 d=s[pn]; u,v=d['u'],d[vn]; candidates={}
 for nm,f in fs.items():
  ks=np.r_[np.linspace(-2,-.01,75),np.linspace(.01,2,75)]; ks=ks[ks>0] if nm=='rat' else ks
  q=min(range(len(ks)),key=lambda i:loss(aff(f(u,ks[i])),v)[0]); k=ks[q]; k=minimize_scalar(lambda k:loss(aff(f(u,k)),v)[0],bounds=(max(1e-5,k-.035) if nm=='rat' or k>0 else k-.035,k+.035 if k>0 else min(-1e-5,k+.035)),method='bounded').x
  A=aff(f(u,k)); B=np.linalg.lstsq(A,v,rcond=None)[0]; candidates[nm]=(loss(A,v),k,aff(f(z,k))@B)
 for deg in [2,3,4,5]:
  A=np.hstack([np.ones((len(u),1))]+[u**j for j in range(1,deg+1)]); B=np.linalg.lstsq(A,v,rcond=None)[0]; candidates['add'+str(deg)]=(loss(A,v),0,np.hstack([np.ones((len(z),1))]+[z**j for j in range(1,deg+1)])@B)
 print('\n',pn,vn,'input-transformed',[(k,tuple(round(i,6) for i in v[0]),round(v[1],5)) for k,v in sorted(candidates.items(),key=lambda q:q[1][0][1])]); outputs[vn]=[min(candidates.values(),key=lambda v:v[0][1])[2]]
 # Nonlinear projection followed by a scaled/shifted output transformation.
 for nm in ['tanh','sin','rat','exp','quad','cubic','softplus']:
  f=fs[nm]; es=[]; pars=[]; preds=[]
  for vv in v.T:
   A=aff(u); lin=np.linalg.lstsq(A,vv,rcond=None)[0]; scale=np.ptp(vv)/2
   def model(p,a):return p[0]+p[1]*f(a@p[2:],1.)
   starts=[np.r_[vv.mean(),scale*k,0,lin[1:]/(scale*k)] for k in (.65,1.3,2.5)]
   sols=[least_squares(lambda p:model(p,A)-vv,p,max_nfev=250) for p in starts]; out=min(sols,key=lambda r:r.cost);es.append(np.sqrt(2*out.cost/len(vv)));pars.append(out.x); preds.append(model(out.x,aff(z)))
  print('out',nm,es,'mean',np.mean(es));
  if np.mean(es)<.02: print(np.array(pars));outputs[vn].append(np.array(preds).T)
 for a in outputs[vn]:
  for name, data in [('panel',a),('panel+raw',np.c_[a,z])]:
   for deg in (1,2):
    A=PolynomialFeatures(deg).fit_transform(data);B=np.linalg.solve(A[:52].T@A[:52]+.003*np.eye(A.shape[1]),A[:52].T@Y[:52]); E=abs(A[52:]@B-Y[52:]);print('CV',name,deg,E[:,0].mean(),E[:,1:].mean())
# Raw sparse structure and two-mode fit.
for end in [52,72]:
 p=PolynomialFeatures(3);A=p.fit_transform(z); B=np.linalg.lstsq(A[:end],Y[:end],rcond=None)[0];print('cubic coefficients',end,np.c_[p.get_feature_names_out(),np.round(B[:,[0,1,8]],4)]); print('cubic training RMSE',np.sqrt(np.mean((A[:end]@B-Y[:end])**2,0)))
U,S,V=np.linalg.svd(Y[:52,1:],full_matrices=False)
for names in [('exp','exp'),('exp','sin'),('exp','linear'),('exp','decay')]:
 def bas(q):
  def fn(k,p):return {'exp':lambda: -np.expm1(-np.exp(p)*t),'sin':lambda:np.sin(np.exp(p)*t),'linear':lambda:t,'decay':lambda:np.exp(-np.exp(p)*t)}[k]()
  return np.array([fn(k,p) for k,p in zip(names,q)])
 def fun(q):
  F=bas(q);return (Y[:52,1:]@np.linalg.pinv(F)@F-Y[:52,1:]).ravel()
 r=least_squares(fun,np.log([.5,2]),bounds=(-5,3));print('time basis',names,np.exp(r.x),np.sqrt(np.mean(r.fun**2)),bas(r.x))
