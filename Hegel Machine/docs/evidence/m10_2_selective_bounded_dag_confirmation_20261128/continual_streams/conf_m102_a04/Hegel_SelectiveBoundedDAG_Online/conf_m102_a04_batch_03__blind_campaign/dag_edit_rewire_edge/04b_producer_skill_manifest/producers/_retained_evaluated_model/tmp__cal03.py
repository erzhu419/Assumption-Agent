exec(open('/tmp/inspect03.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
from itertools import combinations
p=s['panel_a']; u,v=p['u'],p['v']; print('shapes',[(k,{kk:np.shape(vv) for kk,vv in s[k].items()}) for k in ('panel_a','panel_b','panel_c')]); print('study keys',s.keys()); F=np.c_[np.ones(len(u)),u]; rng=np.random.default_rng(1); Q=[]
for j in range(4):
 r=np.sort(np.abs(v[:,j]-np.c_[np.ones(len(u)),u[:,j]]@np.linalg.lstsq(np.c_[np.ones(len(u)),u[:,j]],v[:,j],rcond=None)[0])); print('diag resid',j,r)
 sols=[]
 for i in range(3000):
  ii=rng.choice(len(u),5,replace=False); q=np.linalg.lstsq(F[ii],v[ii,j],rcond=None)[0]; r=abs(F@q-v[:,j]); sols.append((sum(r<.001),np.median(r),q))
 count,med,q=min(sols,key=lambda x:(-x[0],x[1])); good=abs(F@q-v[:,j])<.001; q=np.linalg.lstsq(F[good],v[good,j],rcond=None)[0]; print('full affine',count,med,q); Q.append(q)
Q=np.array(Q).T
p=s['panel_b']; B=np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0]; print('B',B,'sv',np.linalg.svd(B[1:],compute_uv=False));
p=s['panel_c']; u=p['u'];
def solve_c(u,key):
 def basis(q):
  if key=='v1': return np.c_[np.ones(len(u)),u/(1+q[0]*abs(u)),u]
  h=u*u; return np.c_[np.ones(len(u)),h,h/(1+q[0]*h)]
 def res(q):
  f=basis(q); return (f@np.linalg.lstsq(f,p[key],rcond=None)[0]-p[key]).ravel()
 opt=least_squares(res,[.5],bounds=(.01,10),gtol=1e-13); f=basis(opt.x); return opt.x,np.mean(abs(res(opt.x))),np.linalg.lstsq(f,p[key],rcond=None)[0]
for name,x in [('u',u),('b',u@B[1:]+B[0]),('binv',(u-B[0])@np.linalg.inv(B[1:]))]:
 for k in ('v1','v2'): print('panel_c',name,k,solve_c(x,k))
A=b.run_node({'__input__':s})['basis']; Y=np.c_[tr['y'],tr['traj']@np.linalg.pinv(A).T]; off,scale=a._align(s['panel_a']); x=tr['x'];
for name,z in [('diag',(x-off)/scale),('full',(x-Q[0])@np.linalg.pinv(Q[1:])),('raw',x)]:
 for bn,w in [('',z),('B',z@B[1:]+B[0]),('Bi',(z-B[0])@np.linalg.inv(B[1:]))]:
  print(name+bn,'ranges',np.quantile(w,[0,1],axis=0));
  for desc,F in [('poly2',PolynomialFeatures(2).fit_transform(w)),('poly3',PolynomialFeatures(3).fit_transform(w)),('rational',np.c_[PolynomialFeatures(2).fit_transform(w),w/(1+.8*abs(w)),w*w/(1+.25*w*w),np.sin(w),np.cos(w)])]:
   pred=F[52:]@np.linalg.lstsq(F[:52],Y[:52],rcond=1e-8)[0]; d=pred-Y[52:]; print(desc,np.mean(abs(d[:,0])),np.mean(abs(d[:,1:]@A.T)))
