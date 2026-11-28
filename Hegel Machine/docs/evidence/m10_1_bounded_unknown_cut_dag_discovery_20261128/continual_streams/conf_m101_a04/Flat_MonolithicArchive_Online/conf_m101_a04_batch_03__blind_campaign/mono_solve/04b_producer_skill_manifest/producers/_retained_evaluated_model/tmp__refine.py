import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from graph_probe import s,fit,poly
z=np.load('/tmp/branches.npz'); x=z['x']; Y=np.c_[z['y'],z['T']]; p=s['panel_c']; u=p['u']; t=s['eval_times']

def regress(des, freq, bounds, v):
 def res(w):
  a=des(u,w);return (a@fit(a,v)-v).ravel()
 sol=least_squares(res,freq,bounds=bounds,gtol=1e-13,ftol=1e-12,xtol=1e-12)
 w=sol.x;a=des(u,w); b=fit(a,v);r=a@b-v; Q=np.linalg.svd(a,full_matrices=False)[0]; loo=np.mean((r/(1-(Q**2).sum(1))[:,None])**2)**.5
 return np.std(r),loo,w,b,des(x,w)@b

base=lambda a:np.c_[np.ones(len(a)),a]
models={
 'sin':(lambda a,f:np.c_[base(a),np.sin(f[0]*a)],[3.3],[.6],[6]),
 'sin+sin':(lambda a,f:np.c_[base(a),np.sin(f[0]*a),np.sin(f[1]*a)],[3.3,5.8],[.7,3.6],[4.8,9]),
 'sin+cos':(lambda a,f:np.c_[base(a),np.sin(f[0]*a),np.cos(f[1]*a)],[3.3,1.3],[.4,.1],[6,9]),
 'sin+cubic':(lambda a,f:np.c_[base(a),np.sin(f[0]*a),a**3],[3.3],[.3],[7]),
 'sin+soft':(lambda a,f:np.c_[base(a),np.sin(f[0]*a),a/(1+f[1]*abs(a))],[3.3,1.],[.3,.05],[7,10]),
 'sin+quad':(lambda a,f:np.c_[base(a),np.sin(f[0]*a),a*a],[3.3],[.3],[7]),
 'cos':(lambda a,f:np.c_[base(a),np.cos(f[0]*a)],[1.26],[.1],[6]),
 'cos+cos':(lambda a,f:np.c_[base(a),np.cos(f[0]*a),np.cos(f[1]*a)],[1.26,2.6],[.3,1.6],[2.,9]),
 'cos+quad':(lambda a,f:np.c_[base(a),np.cos(f[0]*a),a*a],[1.26],[.2],[6]),
 'cos+cubic':(lambda a,f:np.c_[base(a),np.cos(f[0]*a),a**3],[1.26],[.2],[6]),
 'cos+cross':(lambda a,f:np.c_[base(a),np.cos(f[0]*a),a*np.roll(a,-1,axis=1)],[1.26],[.2],[6]),
}
branches={}
for j,vname in enumerate(('v1','v2')):
 for name,(des,freq,lo,hi) in models.items():
  if j==0 and not name.startswith('sin') or j==1 and not name.startswith('cos'):continue
  e,cv,w,b,out=regress(des,freq,(lo,hi),p[vname]);print(vname,name,'rms',round(e,7), 'panel loo',round(cv,7),'freq',w,'block norms',[round(np.linalg.norm(r),5) for r in np.array_split(b[1:],len(b[1:])//4)]);branches[(vname,name)]=out

def check(F, n=52):
 pred=F[n:72]@fit(F[:n],Y[:n]); return np.round(abs(pred-Y[n:]).mean(0)[[0,-1]],5).tolist()+[round(abs(pred[:,1:]-Y[n:,1:]).mean(),5)]
for (v1,cn),c in branches.items():
 if v1!='v1':continue
 for (v2,dn),d in branches.items():
  if v2!='v2':continue
  g=(c[:,:,None]*d[:,None,:]).reshape(-1,16)
  print(cn,dn,check(np.c_[base(c),d,g]))
for cn,dn in [('sin','cos'),('sin+sin','cos'),('sin+soft','cos')]:
 c,d=branches['v1',cn],branches['v2',dn];g=(c[:,:,None]*d[:,None,:]).reshape(-1,16);f=np.c_[base(c),d,g];w=fit(f[:52],Y[:52]);print('\nweights',cn, np.round(w[:,[0,1,-1]],3));print('diag check',check(np.c_[base(c),d,c*d]));np.savez('/tmp/refined_'+cn+'.npz',x=x,c=c,d=d,Y=Y)
T=Y[:52,1:]; print('times',t)
for kind,fn,w0 in [('exp+exp',lambda w:np.stack([-np.expm1(-w[0]*t),-np.expm1(-w[1]*t)]),[.25,1.3]),('exp+log',lambda w:np.stack([-np.expm1(-w[0]*t),np.log1p(w[1]*t)]),[1.3,1]),('exp+t',lambda w:np.stack([-np.expm1(-w[0]*t),t]),[1.3]),('exp+rat',lambda w:np.stack([-np.expm1(-w[0]*t),t/(1+w[1]*t)]),[1.3,1]),('exp+sin',lambda w:np.stack([-np.expm1(-w[0]*t),np.sin(w[1]*t)]),[1.3,.5])]:
 def obj(w):
  a=fn(w).T;return (a@fit(a,T.T)-T.T).ravel()
 w=least_squares(obj,w0,bounds=(.0001,6)).x; print(kind,w,'temporal rms',np.std(obj(w)))
