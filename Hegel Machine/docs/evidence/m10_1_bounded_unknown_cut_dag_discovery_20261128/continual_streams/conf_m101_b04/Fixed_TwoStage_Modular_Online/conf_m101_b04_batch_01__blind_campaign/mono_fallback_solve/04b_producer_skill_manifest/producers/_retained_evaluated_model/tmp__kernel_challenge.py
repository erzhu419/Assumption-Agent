import importlib.util,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; a=m._calibration(s['panel_a'],True); b=m._calibration(s['panel_b']); z=m._linear(x)@a; w=m._linear(z)@b; bi=np.linalg.inv(b[1:]); ai=np.linalg.inv(a[1:]); zz=((x-b[0])@bi-a[0])@ai
rng=np.random.default_rng(123); folds=[np.arange(72)]+[rng.permutation(72) for _ in range(11)]; folds += [np.argsort(x[:,k]*sign) for k in range(4) for sign in [-1,1]]
def candidates(x,z,w,zz):
 d=cdist(w,w,'sqeuclidean'); dp=(z[:,None]-z[None,:])**2; di=cdist(zz,zz,'sqeuclidean'); cov=(x@x.T)/2; q=(1+cov)**2; c=cov**3; p=m._linear(x); U,V=np.triu_indices(4); P=x[:,U]*x[:,V]; Q=P@P.T
 for scale in [.08,.16,.25,.4,.65]:
  for kind in ['rbf','mat','rq','prod','add','inv']:
   if kind=='mat': h=np.sqrt(5*scale*d); K=(1+h+h*h/3)*np.exp(-h)
   elif kind=='rq': K=(1+scale*d)**-2
   elif kind=='prod': K=np.prod(1/(1+scale*dp),axis=2)
   elif kind=='add': R=np.exp(-scale*dp); K=(R.sum(2)+sum(R[:,:,j]*R[:,:,k] for j in range(4) for k in range(j)))/10
   elif kind=='inv': K=np.exp(-scale*di)
   else: K=np.exp(-scale*d)
   for trend in [0,1,2,3]:
    H=K if trend==0 else K+q if trend==1 else K+q+.15*c if trend==2 else .25*K+q+Q
    for noise in [.0002,.001,.004,.016]:yield (scale,kind,trend,noise),H+noise*np.eye(len(x))
def pred_bank(x,z,w,zz,i,j,target):
 pred=[]; err=[]; tags=[]; Y=target[i]; Y=Y[:,None] if Y.ndim==1 else Y
 for tag,K in candidates(x,z,w,zz):
  f=cho_factor(K[np.ix_(i,i)]); ki=cho_solve(f,np.eye(len(i))); A=ki@Y; e=A/ki.diagonal()[:,None]; pred.append(K[np.ix_(j,i)]@A); err.append(abs(e).mean()); tags.append(tag)
 return np.array(pred),np.array(err),tags
for src in [0,1]:
 E=[]; choices=[]
 for k,f in enumerate(folds):
  i,j=f[:52],f[52:]; o=m.solve_batch(dict(s,train=dict(x=x[i],y=y[i],traj=t[i]),test=dict(x=x[j]))); target=y if src==0 else t; v=target[j]; v=v[:,None] if src==0 else v
  p,l,tags=pred_bank(x,z,w,zz,i,j,target); rank=np.argsort(l); p0=o['y_hat'][:,None] if src==0 else o['state_pred']; choices.append((tags[rank[0]],round(l.min(),4))); ps=[p0,p[rank[0]]];
  for temperature in [.03,.08,.18]:
   weight=np.exp(-(l/l.min()-1)/temperature); pt=np.average(p,axis=0,weights=weight); ps.extend([pt,.5*(pt+p0)])
  if src==1:
   mean=t[i].mean(0); modes=np.linalg.svd(t[i]-mean,full_matrices=False)[2][:2]; ps=[mean+((p-mean)@modes.T)@modes for p in ps]
  E.append(np.mean(np.abs(np.array(ps)-v),axis=(1,2)))
 E=np.array(E); print('src',src,'cols=baseline,loo winner,temp and blend pairs'); print(E[:12].round(5)); print('rand mean/worst',E[:12].mean(0).round(5),E[:12].max(0).round(5),'boundary mean/worst',E[12:].mean(0).round(5),E[12:].max(0).round(5)); print('worst random alternatives',choices[9]); print('choices',choices[:12])
