exec(open('/tmp/probe_walls.py').read().split('for walls in itertools.product')[0])
from scipy.optimize import least_squares
from scipy import sparse
from scipy.sparse.linalg import splu
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]
rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
I=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); axes=[]
for axis,n in [('x',nx),('y',ny)]:
 h=s['L'+axis]/(n-1); k=s['kappa_'+axis]; v=s['vel_'+axis]; b=s['robin_'+axis]
 a=prior._axis(k,v,h,b,'sealed'); p=prior._axis(k,v,h,b,'periodic'); r=prior._axis(k,v,h,b,'robin'); axes.append((a,p-a,r-a))
def forward(par):
 ax,ay=(a+t*d+l*r for (a,d,r),(t,l) in zip(axes,np.reshape(par,(2,2))))
 L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])
center=np.repeat(s['total_budget']/rhs.shape[1],rhs.shape[1]); N=np.linalg.qr(np.ones((len(center),1)),mode='complete')[0][:,1:]
def residual(S,idx,info=False):
 D=np.column_stack((S@N,z)); target=y-S@center; coeff=np.linalg.lstsq(D[idx],target[idx],rcond=None)[0]; w=center+N@coeff[:N.shape[1]]; e=target-D@coeff
 return (e,w) if info else e[idx]
allrows=np.ones(len(y),bool)
for a,b in itertools.product((0.,1.),repeat=2):
 e,w=residual(forward([a,0,b,0]),allrows,True); print('discrete wrap',a,b,'RMSE',np.sqrt(np.mean(e**2)),'w range',(w.min(),w.max()))
lo=[0,0,0,0]; hi=[1.5,0.7,1.5,0.7]
def fitpar(idx): return least_squares(lambda p: residual(forward(p),idx),[.8,.01,.8,.01],bounds=(lo,hi),gtol=1e-10).x
p=fitpar(allrows); print('fitted wrap/leak',p,'RMSE',np.sqrt(np.mean(residual(forward(p),allrows)**2)))
for name,g in [('rosette',groups),('shift',shifts),('bank',s['row_bank_id'])]:
 for label,par in [('PP',[1,0,1,0]),('SP',[0,0,1,0]),('fitted',None)]:
  vals=[]; parms=[]
  for j in np.unique(g):
   idx=g!=j; p=fitpar(idx) if par is None else par; parms.append(p); e,_=residual(forward(p),idx,True); vals.append(np.sqrt(np.mean(e[~idx]**2)))
  print(name,label,'aggregate/worst',np.sqrt(np.mean(np.square(vals))),max(vals),'parameter ranges',np.min(parms,axis=0),np.max(parms,axis=0))
