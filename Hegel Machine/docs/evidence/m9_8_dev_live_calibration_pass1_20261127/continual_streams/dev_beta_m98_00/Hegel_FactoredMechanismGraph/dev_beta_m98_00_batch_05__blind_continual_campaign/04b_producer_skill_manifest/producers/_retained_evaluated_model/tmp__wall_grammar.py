exec(open('/tmp/wall_probe.py').read().split('allrows=')[0])
from scipy.sparse import lil_matrix
orig=m._axis
counts=np.arange(len(y)); sensor=counts%len(s['sensor_ij']); rosette=sensor//4; shift=counts//len(s['sensor_ij']); full=np.ones(len(y),bool)
folds=[rosette%4!=j for j in range(4)]+[shift!=j for j in range(len(s['mu_shifts']))]
# Wall row changes: leakage, conservative nearest/next-nearest flux, and upwind inflow leakage.
B={
 'xhi robin':[(0,1,'leak',1)], 'xhi face':[(0,1,'face',1)], 'xhi next':[(0,1,'next',1)],
 'xhi row scale':[(0,1,'leak',1),(0,1,'face',1)], 'yhi robin':[(1,1,'leak',1)],
 'yhi face':[(1,1,'face',1)], 'yhi next':[(1,1,'next',1)],
 'paired high robin':[(0,1,'leak',1),(1,1,'leak',1)],
 'all robin':[(a,b,'leak',1) for a in range(2) for b in range(2)],
 'all wall row scale':[(a,b,k,1) for a in range(2) for b in range(2) for k in ('leak','face')],
 'inflow advective leakage':[(a,b,'inflow',1) for a in range(2) for b in range(2)],
 'outflow advective leakage':[(a,b,'outflow',1) for a in range(2) for b in range(2)],
}
def A(law,p):
 def axis(k,v,h,robin,wall):
  g=orig(k,v,h,robin,wall).tolil(); a=0 if len(k)==s['nx'] else 1
  for aa,b,term,sgn in law:
   if aa!=a: continue
   i,j,j2=(0,1,2) if b==0 else (len(k)-1,len(k)-2,len(k)-3)
   if term=='leak': g[i,i]+=p*sgn*robin/h
   if term=='face': g[i,i]+=p*sgn*k[i]/h**2; g[i,j]-=p*sgn*k[i]/h**2
   if term=='next': g[i,j]+=p*sgn*k[i]/h**2; g[i,j2]-=p*sgn*k[i]/h**2
   if term in ('inflow','outflow'):
    u=v[i]*(1-2*b)*(1 if term=='inflow' else -1); g[i,i]+=p*sgn*max(0,u)/h
  return g.tocsc()
 m._axis=axis
 return m._response(s,np.ones(8),modes)[0]
def fit(law, rows):
 zr=z[rows]; f=np.linalg.lstsq(zr,np.eye(rows.sum()),rcond=None)[0]; target=y[rows]-zr@(f@y[rows]); r=1e-7
 def calc(p,report=False):
  a=A(law,p[0]); aa=a[rows]; x=aa-zr@(f@aa); w=m._weights(x,target,r,cap,total)
  if not report:return np.r_[target-x@w,np.sqrt(r)*w]
  err=y-a@w-z@(f@(y[rows]-aa@w)); return np.sqrt(np.mean(err[~rows if not rows.all() else rows]**2))
 p=least_squares(calc,[0.],bounds=(-.95,3),max_nfev=25).x if law else [0.]
 return float(p[0]),calc(p,True)
for name,law in [('nominal',[])]+list(B.items()):
 p,rm=fit(law,full); vals=np.array([fit(law,rows) for rows in folds]); print(name,round(p,3),round(rm,6),'held out',np.round(vals[:,1],6),'fold parameters',np.round(vals[:,0],2))
