import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def one(u):return np.c_[np.ones(len(u)),u]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
ix,jx=np.triu_indices(4,1)
def prod(u):return u[:,ix]*u[:,jx]
def prof(fn,u,v,q,bounds):
 def r(q):
  f=fn(u,q); return (f@ls(f,v)-v).ravel()
 q=least_squares(r,q,bounds=bounds,gtol=1e-12).x
 return q,ls(fn(u,q),v),np.std(r(q))
def base(s,x):
 p=s['panel_a'];keep=np.linalg.norm(p['v'],axis=1)<4;u,v=p['u'][keep],p['v'][keep]
 def f(u,q):return np.stack([np.ones_like(u),u,np.tanh(q[0]*u),np.sin(q[1]*np.roll(u,2,axis=-1))],-1)
 def fit(q):
  a=f(u,q);w=np.array([ls(a[:,i],v[:,i]) for i in range(4)]);return w,(a*w).sum(-1)-v
 q=least_squares(lambda q:fit(q)[1].ravel(),[1.8,2.4],gtol=1e-12).x;w,r=fit(q)
 z=np.array([least_squares(lambda z:(f(z,q)*w).sum(-1)-row,(row-w[:,0])/(w[:,1]+w[:,2]),gtol=1e-12).x for row in x]);print('calibration',q,'error',r.std(),'weights',w,'z range',z.min(),z.max());return z
p=s['panel_c'];u=p['u'];z=base(s,np.r_[s['train']['x'],s['test']['x']]);v=p['v1'];hfn=lambda w,q:one(np.c_[w,np.tanh(q[0]*w),prod(w)])
q,w,r=prof(hfn,u,p['v2'],[1.2],(.3,4));print('H',q,r,np.around(w,5));h=hfn(z,q)@w
b=one(z)@ls(one(s['panel_b']['u']),s['panel_b']['v'])
opts={
 'soft+linear':(lambda x,q:one(np.c_[x,x/(1+q[0]*abs(x))]),[.8],(.01,7)),
 'softmix':(lambda x,q:one(x/(1+q[0]*abs(x))+q[1]*x),[.8,.03],([.01,-1],[7,1])),
 'tanh+linear':(lambda x,q:one(np.c_[x,np.tanh(q[0]*x)]),[1.7],(.1,5)),
 'tanh+linear+p':(lambda x,q:one(np.c_[x,np.tanh(q[0]*x),prod(x)]),[1.7],(.1,5)),
 'tanh+p':(lambda x,q:one(np.c_[np.tanh(q[0]*x),prod(x)]),[1.7],(.1,5)),
 'tanh+sat':(lambda x,q:one(np.c_[np.tanh(q[0]*x),x/(1+q[1]*abs(x))]),[1.7,.7],([.1,.01],[6,6])),
 'tanh+sin':(lambda x,q:one(np.c_[np.tanh(q[0]*x),np.sin(q[1]*x)]),[1.7,2.4],(.1,6)),
 'rat+p':(lambda x,q:one(np.c_[x,x/(1+(q[0]*x)**2),prod(x)]),[1.2],(.1,5)),
 'tanh+linear+cube':(lambda x,q:one(np.c_[x,np.tanh(q[0]*x),x**3]),[1.7],(.1,5)),
 }
Y=np.c_[s['train']['y'],s['train']['traj']];res=[]
for name,(fn,q0,bd) in opts.items():
 q,w,r=prof(fn,u,v,q0,bd);g=fn(z,q)@w;print('G',name,q,r)
 if name in ['soft+linear','tanh+linear','tanh+p']:print(np.around(w,4))
 for n,a in {'gh':np.c_[g,h], 'bgh':np.c_[b,g,h], 'bh':np.c_[b,h],'bg':np.c_[b,g]}.items():
  for nn,f in {'linear':a,'dot':np.c_[a,(g*h).sum(1)],'diag':np.c_[a,g*h],'cross':np.c_[a,cross(g,h)],'all':np.c_[a,cross(a,a)[:,np.triu_indices(a.shape[1])[0]*a.shape[1]+np.triu_indices(a.shape[1])[1]]], 'binter':np.c_[a,cross(b,g),cross(b,h)],'bxg':np.c_[a,cross(b,g)],'bxh':np.c_[a,cross(b,h)]}.items():
   f=one((f-f[:52].mean(0))/np.maximum(f[:52].std(0),1e-8));f0=f[:52]
   for reg in [0,.001,.03,.3]:
    coef=ls(f0,Y[:52]) if reg==0 else np.linalg.solve(f0.T@f0+reg*np.eye(f.shape[1]),f0.T@Y[:52]);err=np.abs(f[52:72]@coef-Y[52:]);res.append((err[:,0].mean(),err[:,1:].mean(),name,n,nn,reg))
for k in [0,1]: print('TOP',k,*sorted(res,key=lambda x:x[k])[:12],sep='\n')
