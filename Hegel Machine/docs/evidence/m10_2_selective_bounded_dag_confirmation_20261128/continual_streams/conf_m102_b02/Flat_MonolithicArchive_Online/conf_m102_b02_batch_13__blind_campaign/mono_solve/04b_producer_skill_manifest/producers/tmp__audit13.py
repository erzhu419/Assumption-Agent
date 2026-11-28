import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=m._calibrate(s['panel_a'],tr['x']); I=m._intercept; ls=m._ls
rms=lambda e: np.sqrt(np.mean(e**2)); p=s['panel_b']; u,v=p['u'],p['v'];
for name,fn,qs in [('cos',np.cos,[2.6]),('sin',np.sin,[1,3,6,10,13]),('rational even',lambda a:1/(1+a*a),[.4,1,2,4]),('quadratic',np.square,[1])]:
 def fun(q):
  d=I(np.c_[u,fn(q[0]*u)]);return (d@ls(d,v)-v).ravel()
 opt=min((least_squares(fun,[q],bounds=(.1,15)) for q in qs),key=lambda a:a.cost);print('panel B',name,opt.x,rms(opt.fun))
w=m._operator(p,'v',lambda a,q:I(np.cos(q*a)),2.6,(1.5,3.5),z); p=s['panel_c']; u,v=p['u'],p['v1']; models=[]
for name,fn,q0,lo,hi in [('softsign',lambda a,q:a/(1+q[0]*abs(a)),[.83],[.05],[8]),('rational',lambda a,q:a/(1+q[0]*a*a),[1],[.05],[20]),('sqrt',lambda a,q:a/np.sqrt(1+q[0]*a*a),[2],[.05],[30]),('asinh',lambda a,q:np.arcsinh(q[0]*a),[3],[.05],[20]),('arctan',lambda a,q:np.arctan(q[0]*a),[2],[.05],[40]),('softsign generalized',lambda a,q:a/(1+q[0]*abs(a)**q[1])**q[2],[1,1,1],[.05,.3,.1],[12,3,5]),('odd quotient',lambda a,q:a/(1+q[0]*abs(a)+q[1]*a*a),[.8,.2],[.01,-.3],[8,8])]:
 for lin in (False,True):
  def d(a,q):return I(np.c_[a,fn(a,q)] if lin else fn(a,q))
  def fit(i):
   def residual(q):
    a=d(u[i],q);return (a@ls(a,v[i])-v[i]).ravel()
   q=least_squares(residual,q0,bounds=(lo,hi),max_nfev=90,ftol=1e-11).x
   return q,ls(d(u[i],q),v[i])
  folds=np.array_split(np.random.default_rng(14).permutation(len(u)),4); errors=[]
  for j in folds:
   q,b=fit(np.setdiff1d(np.arange(len(u)),j));errors.append(d(u[j],q)@b-v[j])
  q,b=fit(np.arange(len(u)));models.append((name,lin,rms(np.concatenate(errors)),d(w,q)@b)); print('panel C',name,lin,np.round(q,5),rms(d(u,q)@b-v),models[-1][2]);
y=np.c_[tr['y'],tr['traj']]; order=np.arange(len(z)); rng=np.random.default_rng(1745); splits=[order]+[rng.permutation(len(z)) for _ in range(12)]+[np.argsort(sign*z[:,j]) for j in range(4) for sign in (-1,1)]
for name,lin,cv,h in sorted(models,key=lambda a:a[2])[:6]:
 base=I(np.c_[z,h,(z[:,:,None]*h[:,None,:]).reshape(len(z),-1)]); pairs=np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]+[h[:,i]*h[:,j] for i in range(4) for j in range(i,4)]); qs=[.25,1,4]; rat=np.column_stack([col/(1+q*delta**2) for col in h.T for q in qs for delta in (z[:,0]-z[:,1],z[:,2]-z[:,3])]);
 for label,extra in [('bilinear',None),('pair products',pairs),('difference quotients',rat)]:
  X=base if extra is None else np.c_[base,extra]; scores=[]
  for p in splits:
   i,j=p[:52],p[52:]; _,_,vh=np.linalg.svd(y[i,1:],full_matrices=False); target=np.c_[y[i,0],y[i,1:]@vh[:2].T]; pred=X[j]@ls(X[i],target); e=abs(np.c_[pred[:,0],pred[:,1:]@vh[:2]]-y[j]);scores.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(axis=0).max()]);
  print('readout',name,lin,label,'required',np.round(scores[0],6),'worst',np.round(np.max(scores,axis=0),6));
print('study keys',list(s),'train keys',list(tr),'telemetry shape',s['telemetry_vec'].shape)
