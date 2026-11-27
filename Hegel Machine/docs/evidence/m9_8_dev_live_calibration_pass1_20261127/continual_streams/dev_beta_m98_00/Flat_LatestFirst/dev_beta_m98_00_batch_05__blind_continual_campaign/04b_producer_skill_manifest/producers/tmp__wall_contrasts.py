exec(open('/tmp/wall_diag.py').read().split("F={'base'")[0])
np.set_printoptions(precision=6,suppress=True)
loc=np.arange(len(y))%len(s['sensor_ij']); shift=np.arange(len(y))//len(s['sensor_ij']); ij=s['sensor_ij'];
cv=[[(folds==k) for k in range(4)],[(shift==k) for k in range(3)],[(ij[loc,0]==0),(ij[loc,1]==0),(ij[loc,0]>=10),(ij[loc,1]>=8)]]
F={'nominal':[], 'common':[[1,1,1,1]], 'x only':[[1,1,0,0]], 'y only':[[0,0,1,1]], 'x versus y':[[1,1,-1,-1]], 'x end contrast':[[1,-1,0,0]], 'y end contrast':[[0,0,1,-1]], 'both end contrast':[[1,-1,1,-1]], 'cross end contrast':[[1,-1,-1,1]]}
for name,g in F.items():
 def estimate(mask):
  def evaluate(t,ret=False):
   wall=np.ones(8); u=np.asarray(t)@np.asarray(g) if g else np.zeros(4); wall[[0,1,4,5]]=np.exp(u)
   a=np.c_[m._response(s,wall,src)[0],z]; q=lsq_linear(np.vstack((a[mask],10*np.r_[np.ones(n),np.zeros(nz)])),np.r_[y[mask],10*total],bounds=bounds,method='bvls',tol=1e-11).x
   r=a@q-y; return (np.asarray(t),r) if ret else r[mask]
  t=least_squares(evaluate,np.zeros(len(g)),bounds=(-1,1),ftol=1e-9,gtol=1e-10).x if g else []
  return evaluate(t,True)
 t,r=estimate(allrows); print(name,'log contrast',t,'train RMS',np.mean(r*r)**.5,flush=True)
 for masks in cv:
  pred=[estimate(~h) for h in masks]; print('  RMS',[round(float(np.sqrt(np.mean(e[h]**2))),6) for (_,e),h in zip(pred,masks)],'log contrasts',[v.round(3).tolist() for v,e in pred],flush=True)
