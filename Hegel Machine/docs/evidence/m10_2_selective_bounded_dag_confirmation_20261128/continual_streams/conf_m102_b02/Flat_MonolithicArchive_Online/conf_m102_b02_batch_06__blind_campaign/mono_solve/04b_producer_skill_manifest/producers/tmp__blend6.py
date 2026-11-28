import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import _coords,_estimate,_i,_cross,_ridge,_response,_operator
import numpy as np,pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; y,tr=a['y'],a['traj']; n=len(y)
z,w,q,r=_coords(s,a['x'],'sat','c'); zs,ws,qs,rs=_coords(s,a['x'],'sine','c'); c=s['panel_c']; rs=_operator(c['u'],c['v2'],ws,lambda x,p:_i(x+p[0]*(1-np.cos(p[1]*x))),[.72,1.2]);
D=[_i(np.c_[z,q,r,_cross(z,q),_cross(q,r)]),_i(np.c_[zs,qs,rs,(zs[:,:,None]*qs[:,None,:]).reshape(n,-1)])]
def estimate(d,z,q,i,j,k,alpha):
 t=tr[i];m=t.mean(0);v=np.linalg.svd(t-m,full_matrices=False)[2][:k];t=(t-m)@v.T;p=_ridge(d[i],t,d[j],alpha);return _response(_i(np.c_[z[i],q[i],t]),y[i],_i(np.c_[z[j],q[j],p]),np.tanh),p@v+m
folds=[('official',np.arange(n))]
for seed in [229,10127]:
 rg=np.random.default_rng(seed);folds.extend((str(seed),rg.permutation(n)) for _ in range(8))
for key in ['z norm','w norm','q norm']:
 x={'z norm':z,'w norm':w,'q norm':q}[key];folds.append((key,np.argsort(np.linalg.norm(x,axis=1))))
errs={}
for group,f in folds:
 i,j=f[:52],f[52:]; base,bt=_estimate(z,w,q,r,y,tr,i,j,fn=np.tanh); p1,t1=estimate(D[0],z,q,i,j,3,.03);p2,t2=estimate(D[1],zs,qs,i,j,1,.005)
 cand={'baseline':(base,bt),'sparse':(p1,t1),'outer':(p2,t2),'sparse_blend':(.3*base+.7*p1,.75*bt+.25*t1),'all_blend':(.25*base+.5*p1+.25*p2,.75*bt+.25*t2),'balanced':(.4*base+.3*p1+.3*p2,.6*bt+.4*t2)}
 for name,(p,t) in cand.items():errs.setdefault((name,group),[]).append([abs(p-y[j]).mean(),abs(t-tr[j]).mean(),np.max(abs(t-tr[j]).mean(0))])
for (name,group),vals in errs.items():
 vals=np.asarray(vals);print(name,group,'avg',vals.mean(0).round(5),'max',vals.max(0).round(5))
