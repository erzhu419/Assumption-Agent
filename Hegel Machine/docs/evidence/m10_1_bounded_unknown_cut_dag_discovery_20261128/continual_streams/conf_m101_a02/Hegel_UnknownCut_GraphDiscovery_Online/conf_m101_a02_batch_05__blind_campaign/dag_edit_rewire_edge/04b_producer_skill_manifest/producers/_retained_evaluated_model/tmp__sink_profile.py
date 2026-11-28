exec(open('/tmp/sink_probe.py').read().split('for label,A in')[0])
from scipy.optimize import least_squares
ids=np.arange(N); y=Y[:,0]; w=x[:,2]-z[:,3]
vs=[ids[52:]]+[f for seed in (7,41,113,307) for f in np.array_split(np.random.default_rng(seed).permutation(ids),4)]+list(np.array_split(ids,4))+[np.argsort(v)[::d][:12] for v in np.c_[x,z].T for d in (1,-1)]
base=np.c_[np.ones(N),z,z[:,0]*z[:,2],z[:,1]*z[:,3]]
def checks(name, fn):
 vals=[]; details=None
 for te in vs:
  tr=np.setdiff1d(ids,te); D,p=fn(tr); c=np.linalg.lstsq(D[tr],y[tr],rcond=None)[0]; vals.append(np.mean(abs(D[te]@c-y[te])))
  if details is None: details=(np.round(p,3),np.round(c,3),round(np.mean(abs(D[tr]@c-y[tr])),5))
 print(name,np.round([vals[0],max(vals[1:21]),max(vals[21:]),np.mean(vals)],5),details)
checks('fixed sin',lambda tr:(np.c_[base,x,np.sin(w)],[]))
checks('fixed rational+triple',lambda tr:(np.c_[base,x[:,1]*x[:,2]*z[:,0],w/(1+w*w)],[]))
for add_x in (False,True):
 B=np.c_[base,x] if add_x else base
 for form in ('sin','sin+cos','tanh','rational'):
  for free_ratio in (False,True):
   def fit(tr):
    def design(p):
     v=p[0]*(x[:,2]-(p[1] if free_ratio else 1.)*z[:,3]); F={'sin':np.sin(v),'sin+cos':np.c_[np.sin(v),np.cos(v)],'tanh':np.tanh(v),'rational':v/(1+v*v)}[form]; return np.c_[B,F]
    def fun(p):
     D=design(p)[tr]; return D@np.linalg.lstsq(D,y[tr],rcond=None)[0]-y[tr]
    pm=[least_squares(fun,[f,1.] if free_ratio else [f],bounds=([.1,.1],[8.,4.]) if free_ratio else ([.1],[8.]),max_nfev=60) for f in (.7,2.,4.)]; p=min(pm,key=lambda o:o.cost).x; return design(p),p
   checks(f'{add_x} {form} ratio={free_ratio}',fit)
