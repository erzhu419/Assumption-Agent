import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from gp05 import affine,apply,temporal,blocks,predict
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,z=[tr[k] for k in ('x','y','traj')]; t=s['eval_times']

def operator(u,v,quad):
 def design(x,w): return np.c_[np.ones(len(x)),x,*([x*x] if quad else []),np.sin(x*w),np.cos(x*w)]
 def fit(i):
  def err(w):
   f=design(u[i],w); return (f@np.linalg.lstsq(f,v[i],rcond=None)[0]-v[i]).ravel()
  p=min((least_squares(err,np.full(4,k),bounds=(.3,5),max_nfev=120) for k in (1,2,3.5,4.5)),key=lambda p:p.cost)
  f=design(u[i],p.x); b=np.linalg.lstsq(f,v[i],rcond=None)[0]
  return p.x,lambda x: design(x,p.x)@b
 ids=np.arange(len(u)); errs=[]
 for j in ids[:0]:
  _,f=fit(ids[ids!=j]); errs.append(f(u[j:j+1])[0]-v[j])
 p,f=fit(ids); print('operator',quad,p,flush=True)
 return f
p=s['panel_c']; c=operator(p['u'],p['v1'],False); d=operator(p['u'],p['v2'],True); a=affine(s['panel_a'],'v',True); b=affine(s['panel_b'],'v')
q=apply(a,x); r=apply(b,q)
coords=[c(q),d(q),c(r),d(r)]
def base(u,n):
 u=(u-u[:n].mean(0))/np.maximum(u[:n].std(0),1e-7); g=u@u.T/u.shape[1]; dist=cdist(u,u,'sqeuclidean'); return [g,g*g,np.exp(-dist/8)]
folds=[np.arange(len(x))]+[np.random.default_rng(i+135).permutation(len(x)) for i in range(6)]; folds += [np.random.default_rng(i+806).permutation(len(x)) for i in range(4)]
folds += [np.argsort(np.linalg.norm(x,axis=1)),np.argsort(np.min(np.linalg.norm(x[:,None]-x[None,:]+100*np.eye(len(x))[:,:,None],axis=2),axis=1))]
results={}
for ids in folds:
 i,j=ids[:56],ids[56:]; Q=temporal(t,z[i]); K=blocks(s,x[ids],56,'panel')[0]; S=blocks(s,x[ids],56,'smooth')[0]
 extra=[base(h[ids],56) for h in coords]
 targets=[predict(k,y[i])[0][:,0] for k in (K,np.concatenate((K,*extra)),np.concatenate((K,extra[1],extra[3])),S)]
 traj=[predict(k,z[i]@Q)[0]@Q.T for k in (S,np.concatenate((S,extra[1])),np.concatenate((S,*extra)))]
 independent=np.column_stack([predict(np.concatenate((S,extra[1])),(z[i]@Q)[:,h])[0][:,0] for h in range(Q.shape[1])])@Q.T
 targets += [.25*targets[0]+.75*targets[1], .2*targets[3]+.8*targets[1]]
 traj += [independent, .5*(traj[1]+traj[2])]
 ey=[np.abs(p-y[j]).mean() for p in targets]; ez=[(np.abs(p-z[j]).mean(),np.abs(p[:,-1]-z[j,-1]).mean()) for p in traj]
 results.setdefault('target',[]).append(ey);results.setdefault('traj',[]).append(ez)
for k,v in results.items():
 r=np.array(v); print(k,'first',np.round(r[0],6),'mean',np.round(r[:-2].mean(0),6),'worst',np.round(r[:-2].max(0),6),'support-edge',np.round(r[-2:],6),sep='\n')
