from basis_probe import V,eval,np,D,center
from scipy.optimize import differential_evolution
for names in [('k','x4','tr8','t2','a1'),('k','x4','xr8','tr8','t2','a1'),('k','x3','x4','xr8','tr4','tr8','t2','s3','a1')]:
 def error(v):
  w=10**np.r_[0.,v[:-1]]; return eval(list(zip(names,w)),10**v[-1])
 def obj(v):
  e=error(v); return .65*e.max()+.35*e.mean()+2.*max(e[0]-.035,0.)
 r=differential_evolution(obj,[(-4,1)]*(len(names)-1)+[(-6,-1)],seed=5,popsize=8,maxiter=48); print(names,'weights/ridge',np.round(10**r.x,7),'metrics',np.round(error(r.x),5),flush=True)
