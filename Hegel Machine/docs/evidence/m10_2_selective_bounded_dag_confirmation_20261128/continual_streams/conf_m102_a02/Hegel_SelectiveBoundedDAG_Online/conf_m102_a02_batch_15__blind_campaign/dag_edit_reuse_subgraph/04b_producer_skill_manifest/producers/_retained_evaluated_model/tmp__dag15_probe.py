import pickle,glob,runpy,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
m=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*batch_08_solver.py')[0]); ls=m['_ls']
c=m['_fit_calibration'](s['panel_a']); print('A terms:',[(t,np.round(w,5)) for t,w in zip(m['_TERMS'],c) if np.any(w)]); u,v=s['panel_a'].values(); r=np.linalg.norm(m['_design'](u,len(c))@c-v,axis=1); print('A residual quantiles',np.quantile(r,[0,.5,.9,.925,.95,1]));
x=s['train']['x']; q=m['_invert'](x,c); print('inverse residual max',np.max(abs(m['_design'](q,len(c))@c-x)), 'q max',abs(q).max()); u,v=s['panel_b'].values()
def B(x,w):return np.c_[np.ones(len(x)),np.cos(w*x)]
def cost(w):
 f=B(u,w);return np.sum((f@ls(f,v)-v)**2)
w=minimize_scalar(cost,bounds=(2.3,2.9),method='bounded').x; b=B(q,w)@ls(B(u,w),v); print('B shared w, rmse',w,np.sqrt(cost(w)/v.size)); h=b*np.roll(q,-1,axis=1)
y=s['train']['y']; z=s['train']['traj']; n=len(q); one=np.ones(n); g=1/(1+s['telemetry_vec'][2]*q[:,2]**2)
def poly(x,d): return np.column_stack([np.prod(x[:,t],axis=1) for j in range(d+1) for t in combinations_with_replacement(range(x.shape[1]),j)])
def score(name,f):
 if f.ndim==1:f=f[:,None]
 d=[]
 for yy in (y,z):
  full=np.mean(abs(f@ls(f,yy)-yy)); es=[]
  for ids in np.array_split(np.arange(n),4)+[np.arange(52,n)]:
   tr=np.setdiff1d(np.arange(n),ids); es.append(np.mean(abs(f[ids]@ls(f[tr],yy[tr])-yy[ids])))
  d.append((round(full,5),round(max(es),5),round(es[-1],5)))
 print(name,f.shape[1],d)
for name,f in {'q quad':poly(q,2),'b quad':poly(b,2),'b linear':np.c_[one,b],'q b':np.c_[one,q,b],'b cyclic':np.c_[one,b,h],'b sum cyclic':np.c_[one,b,h.sum(1)],'b gated cyclic':np.c_[one,b*g[:,None],h,b[:,3]*q[:,2]**3*g],'b all bq':np.c_[one,b,(b[:,:,None]*q[:,None,:]).reshape(n,-1)],'q b quad':poly(np.c_[q,b],2),'q cos':np.c_[one,q,np.cos(w*q)],'b pair and cyclic':np.c_[poly(b,2),h],'q quad + b quad + cyclic':np.c_[poly(q,2),poly(b,2)[:,1:],h]}.items():score(name,f)
np.savez('/tmp/dag15_decoded.npz',q=q,b=b,w=w,y=y,z=z)
