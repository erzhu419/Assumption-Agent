from wall_probe import *
from itertools import product

def sax(k,v,h,r,p):
 a=axis(k,v,h,r,np.array([[1.,0.],[1.,0.]])).tolil(); n=len(k)
 for t,b,j,side in [(0,n-1,1,0),(n-1,0,n-2,1)]:
  f,g,adv=p[side]; orient=1-2*side
  inward=(f+h*adv*max(-orient*v[t],0))/h**2
  outward=(g+h*adv*max(orient*v[t],0))/h**2
  a[t,j]=-inward; a[t,b]=-outward; a[t,t]=inward+outward
 return a.tocsc()

def response(p):
 ax=sax(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],p[:2]); ay=sax(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],p[2:]); L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
 return np.vstack([splu(L+mu*I).solve(src)[sensor] for mu in s['mu_shifts']]).T

ks=[s['kappa_x'],s['kappa_y']]
base=np.array([[(k[t]+k[j])/2,(k[t]+k[b])/2,0.] for k in ks for t,b,j in [(0,-1,1),(-1,0,-2)]])
sw=s['sensor_calibration_sweeps']; cal=calibrate_upstream(s,pay); print('keys after upstream',list(cal)); print('sweep keys',list(sw))
for q in (s,cal):
 u=q['sensor_calibration_sweeps']; ref=u['ref_rosette']; raw=u['raw_rosette']; a=np.c_[ref,np.ones(len(ref))]; m=np.linalg.lstsq(a,raw,rcond=None)[0]; print('forward fit',m.round(7),'RMSE',rms(a@m-raw))
print('base face pairs',base)
for held in [None,0,1,2]:
 train=np.arange(3)!=held
 for n in [4,8,12]:
  def decode(x):
   p=base.copy(); p[:,:2]*=np.array(x[:8]).reshape(4,2) if n>=8 else np.tile(x[:4],2).reshape(4,2)
   if n==12:p[:,2]=x[8:]
   return p
  f=least_squares(lambda x:(response(decode(x))-y)[train].ravel(),np.r_[np.ones(min(n,8)),np.zeros(max(0,n-8))],ftol=1e-12,gtol=1e-12,xtol=1e-12)
  print('fit',held,n,'params',np.round(f.x,5),'faces',decode(f.x).round(5).tolist(),'errs',rms(response(decode(f.x))-y,1))

def grammar(k,inner,outer,adv):
 return [[{'local':k[t],'mid':(k[t]+k[j])/2,'neighbor':k[j]}[inner],{'local':k[t],'mid':(k[t]+k[b])/2,'neighbor':k[b]}[outer],adv] for t,b,j in [(0,-1,1),(-1,0,-2)]]
options=list(product(['local','mid','neighbor'],['local','mid','neighbor'],[0,1])); specs=list(product(options,repeat=2)); errors=np.array([rms(response(grammar(ks[0],*a)+grammar(ks[1],*b))-y,1) for a,b in specs]);
for j in np.argsort(np.max(errors,1))[:6]: print('discrete',specs[j],errors[j])
for held in range(3):
 j=np.argmin(np.mean(errors[:,np.arange(3)!=held]**2,1)); print('discrete LOO',held,specs[j],errors[j])
