import itertools,sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
from scipy.optimize import minimize
base=m._axis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibrate(s); Z=s['Z_nuisance']; ns=len(s['sensor_ij']); n=len(y); nr=s['nuisance_ridge']; ridge=s['source_ridge']; cap=s['u_box']; bu=s['total_budget']; orig=m._forward
nx,ny=s['nx'],s['ny']; xh=s['Lx']/(nx-1); yh=s['Ly']/(ny-1)
for mds in itertools.product(('oo','cc','oc','co','p'),repeat=2):
 def axis(k,v,h,leak):
  j=int(len(k)!=nx); ms=mds[j]
  if ms!='p': return base(k,v,h,tuple(i if t=='o' else 0. for i,t in zip(leak,ms)))
  a=base(k,v,h,(0.,0.)).tolil()
  for r,j,mx in [(0,-1,1),(-1,0,-2)]:
   left=(r==0); f=(k[r]+k[j])*.5/h**2 + max((1 if left else -1)*v[r],0)/h; g=(k[r]+k[mx])*.5/h**2+max((-1 if left else 1)*v[r],0)/h
   a[r,r]=f+g; a[r,j]=-f; a[r,mx]=-g
  return a.tocsc()
 m._axis=axis; S,U=orig(s); E=[]; EP=[]
 def train(g):
  x=S[g]; z=Z[g]; t=y[g]; ze=np.linalg.solve(z.T@z+nr*np.eye(z.shape[1]),z.T); tx=x-z@(ze@x); ty=t-z@(ze@t)
  w,nu=m._box_simplex(tx.T@tx+ridge*np.eye(tx.shape[1]),-tx.T@ty,cap,bu)
  return w, ze@(t-x@w)
 w,g=train(np.ones(n,bool)); rms=np.sqrt(np.mean((S@w+Z@g-y)**2))
 for a in range(0,ns,4):
  ho=((np.arange(n)%ns)//4)==a//4; wa,ga=train(~ho); E.append(np.sqrt(np.mean((S[ho]@wa+Z[ho]@ga-y[ho])**2)))
 for a in range(n//ns):
  ho=np.arange(n)//ns==a; wa,ga=train(~ho); EP.append(np.sqrt(np.mean((S[ho]@wa+Z[ho]@ga-y[ho])**2)))
 print(mds,'fit',round(rms,6),'rosette RMS/max',round(np.sqrt(np.mean(np.square(E))),6),round(max(E),6),'shift',np.round(EP,6),'terminal norm',round(np.linalg.norm(np.tensordot(w,U,axes=1)),6))
