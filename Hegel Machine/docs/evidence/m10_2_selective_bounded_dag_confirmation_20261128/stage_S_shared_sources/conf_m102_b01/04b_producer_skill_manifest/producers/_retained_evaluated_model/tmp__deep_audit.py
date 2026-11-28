import pickle,sys,numpy as np,itertools
from scipy.optimize import minimize_scalar,least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; n=len(d['x']); w=m._operator(s,d['x']); y=d['y']; T=d['traj']; t=s['eval_times']; ids=np.arange(n)
print('study fields',list(s),'train fields',list(d),'telemetry (one vector)',s['telemetry_vec']); print('operator ranges train / test',np.array([w.min(0),w.max(0)]),np.array([m._operator(s,s['test']['x']).min(0),m._operator(s,s['test']['x']).max(0)]))
ls=m._ls; aff=m._aff
u=s['panel_a']['u'];v=s['panel_a']['v']; x=aff(u);q=ls(x,v)
for _ in range(10):
 ix=np.argsort(np.max(abs(x@q-v),axis=1))[:42];q=ls(x[ix],v[ix]);
keep=np.max(abs(x@q-v),axis=1)<.025;q=ls(x[keep],v[keep]);print('A inlier count',sum(keep),'full affine',q,'inlier RMSE',np.sqrt(np.mean((x[keep]@q-v[keep])**2)))
qs=np.zeros_like(q)
for j in range(4):qs[[0,j+1],j]=ls(x[keep][:,[0,j+1]],v[keep,j]);
print('diagonal RMSE',np.sqrt(np.mean((x[keep]@qs-v[keep])**2)))
angles=list(itertools.combinations_with_replacement(range(4),2)); ij=list(itertools.combinations(range(4),2))
def f(w,g,typ):
 z=np.arcsinh(g*w); p=np.array([z[:,i]*z[:,j] for i,j in angles]).T
 if typ=='latent-only':return np.c_[aff(z),p]
 if typ=='cross-only':return np.c_[aff(w),z,*[z[:,i]*z[:,j] for i,j in ij]]
 if typ=='cubic':return np.c_[aff(w),z,p,*[z[:,i]*z[:,j]*z[:,k] for i,j,k in itertools.combinations(range(4),3)]]
 if typ=='rational-pair':
  h=np.c_[w,*[w[:,i]+sign*w[:,j] for i,j in ij for sign in [-1,1]]];return np.c_[aff(w),h/(1+g*h*h),*[w[:,i]*w[:,j] for i,j in angles]]
 return np.c_[aff(w),z,p]
splits=[(ids[:52],ids[52:])];
for k in range(5):
 order=np.argsort(w[:,k] if k<4 else np.linalg.norm(w,axis=1));splits.extend([(np.setdiff1d(ids,b),b) for b in (order[:12],order[-12:])]);
for typ in ['base','latent-only','cross-only','cubic','rational-pair']:
 scores=[]
 for a,b in splits:
  def err(g):
   h=f(w[a],g,typ); return np.mean((h@ls(h,y[a])-y[a])**2)
  g=minimize_scalar(err,bounds=(.8,7),method='bounded').x;h=f(w,g,typ); scores.append(np.mean(abs(h[b]@ls(h[a],y[a])-y[b])))
 print(typ,'prescribed, worst',scores[0],max(scores));
 if typ=='base':
  a,b=splits[0];g=minimize_scalar(lambda g:np.mean((f(w[a],g,typ)@ls(f(w[a],g,typ),y[a])-y[a])**2),bounds=(.8,7),method='bounded').x; print('scalar coefficients at gain',g,ls(f(w[a],g,typ),y[a]));
for typ in ['free','twoexp','exp_log']:
 scores=[]; noise=[]
 for a,b in splits:
  pred=m._predict(np.r_[w[a],w[b]],T[a],False)
  if typ!='free':
   def ftime(p):return np.c_[-np.expm1(-p[0]*t),-np.expm1(-p[1]*t) if typ=='twoexp' else np.log1p(p[1]*t)]
   def resid(p):
    B=ftime(p);return (B@ls(B,T[a].T)-T[a].T).ravel()
   p=least_squares(resid,[1.8,.4],bounds=(.001,10)).x; B=ftime(p); pred=(B@ls(B,pred.T)).T; noise.append((p,np.std(resid(p))))
  e=abs(pred-T[b]);scores.append([e.mean(),e.mean(0).max(),e[:,-1].mean()]);
 print(typ,'prescribed',scores[0],'worst',np.max(scores,0),'projection',noise[:1]);
