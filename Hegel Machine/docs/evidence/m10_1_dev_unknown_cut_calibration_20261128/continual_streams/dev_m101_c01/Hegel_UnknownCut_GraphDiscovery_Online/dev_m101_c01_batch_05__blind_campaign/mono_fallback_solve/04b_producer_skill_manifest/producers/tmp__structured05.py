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
 for j in ids:
  _,f=fit(ids[ids!=j]); errs.append(f(u[j:j+1])[0]-v[j])
 p,f=fit(ids); print('operator',quad,p,'LOO MAE',np.abs(errs).mean(0),flush=True)
 return f
p=s['panel_c']; c=operator(p['u'],p['v1'],False); d=operator(p['u'],p['v2'],True); a=affine(s['panel_a'],'v',True); b=affine(s['panel_b'],'v')
q=apply(a,x); r=apply(b,q)
coords=[c(q),d(q),c(r),d(r)]
def base(u,n):
 u=(u-u[:n].mean(0))/np.maximum(u[:n].std(0),1e-7); g=u@u.T/u.shape[1]; dist=cdist(u,u,'sqeuclidean'); return [g,g*g,np.exp(-dist/8)]
folds=[np.arange(len(x))]+[np.random.default_rng(i+135).permutation(len(x)) for i in range(6)]; folds += [np.random.default_rng(i+806).permutation(len(x)) for i in range(4)]
results={}
for ids in folds:
 i,j=ids[:56],ids[56:]; Q=temporal(t,z[i]); K=blocks(s,x[ids],56,'panel')[0]; S=blocks(s,x[ids],56,'smooth')[0]
 options={'baseline':(K,S),'C1':(np.concatenate((K,base(coords[0][ids],56))),np.concatenate((S,base(coords[0][ids],56)))),'C2':(np.concatenate((K,base(coords[1][ids],56))),np.concatenate((S,base(coords[1][ids],56)))),'both':(np.concatenate((K,*[base(h[ids],56) for h in coords])),np.concatenate((S,*[base(h[ids],56) for h in coords])))}
 for name,(ky,kz) in options.items():
  pred=predict(ky,y[i])[0][:,0]; state=predict(kz,z[i]@Q)[0]@Q.T; ey=np.abs(pred-y[j]); ez=np.abs(state-z[j]); results.setdefault(name,[]).append([ey.mean(),ez.mean(),ez[:,-1].mean()])
for k,v in results.items(): print(k,'first/mean/worst',np.round(v[0],6),np.round(np.mean(v,0),6),np.round(np.max(v,0),6))
