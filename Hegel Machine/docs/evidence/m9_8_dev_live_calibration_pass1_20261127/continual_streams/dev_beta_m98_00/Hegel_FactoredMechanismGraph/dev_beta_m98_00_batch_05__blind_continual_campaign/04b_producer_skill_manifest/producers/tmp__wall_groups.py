exec(open('/tmp/wall_probe.py').read().split('allrows =')[0].split('allrows=')[0])
def fit_group(basis, rows, ridge, fixed=None):
 zr=z[rows]; F=np.linalg.solve(zr.T@zr+1e-7*np.eye(zr.shape[1]),zr.T); q=lambda x:x-zr@(F@x); yt=q(y[rows]);
 def calc(params,ret=False):
  wall=np.exp(params@basis) if len(basis) else np.ones(8); A=m._response(s,wall,modes)[0]; X=q(A[rows]); w=m._weights(X,yt,ridge,cap,total); r=X@w-yt
  if not ret:return np.r_[r,np.sqrt(ridge)*w]
  err=y-A@w-z@(F@(y[rows]-A[rows]@w)); return wall, w, np.sqrt(np.mean(err[~rows]**2)) if not rows.all() else np.sqrt(np.mean(r**2))
 p=least_squares(calc,np.zeros(len(basis)),bounds=(-3,3),max_nfev=35).x if len(basis) else []
 return calc(p,True)
I=np.eye(8); labels={'base':[], 'x+':I[[1]], 'y-':I[[4]], 'y+':I[[5]], 'y symmetric':[I[4]+I[5]], 'x symmetric':[I[0]+I[1]], 'all symmetric':[I[[0,1,4,5]].sum(0)], 'x+/y- paired':[I[1]+I[4]], 'x+/y+ paired':[I[1]+I[5]], 'x+/y- opposed':[I[1]-I[4]], 'x+/y+ opposed':[I[1]-I[5]], 'x+/y± independent':I[[1,4,5]], 'x+/y symmetric':[I[1],I[4]+I[5]], 'x+/y antisymmetric':[I[1],I[4]-I[5]]}
for name,B in labels.items():
 for r in (1e-7,s['source_ridge']):
  wall,w,rmse=fit_group(B,np.ones(len(y),bool),r); rosettes=[fit_group(B,np.arange(len(y))//4%4!=j,r)[-1] for j in range(4)]; shifts=[fit_group(B,np.arange(len(y))//32!=j,r)[-1] for j in range(3)]; print(name,r,'wall',wall[[0,1,4,5]],'fit',round(rmse,6),'rosette',np.round(rosettes,6),'shift',np.round(shifts,6))
