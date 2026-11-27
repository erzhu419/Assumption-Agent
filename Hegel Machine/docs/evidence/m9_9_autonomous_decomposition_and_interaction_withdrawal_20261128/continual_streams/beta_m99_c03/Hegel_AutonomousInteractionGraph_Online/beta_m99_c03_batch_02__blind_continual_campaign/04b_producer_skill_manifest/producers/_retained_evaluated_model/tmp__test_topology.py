from probe_boundary import *
from itertools import product

def op(a,kind):
 k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1); r=s['robin_'+a]; n=len(k)
 if kind=='open': return axis(k,v,h,r,[1,1],[1,1])
 if kind=='reflect': return axis(k,v,h,r,[0,0],[1,1])
 if kind=='mirror': return axis(k,v,h,r,[0,0],[2,2])
 if kind=='absorb': return axis(k,v,h,r,[k[0]/(h*r),k[-1]/(h*r)],[1,1])
 A=axis(k,v,h,r,[0,0],[1,1]).tolil()
 for j,m,b in [(0,n-1,1),(n-1,0,n-2)]:
  if kind=='periodic_endpoint':
   out=k[j]/h**2; inside=out
  else:
   out=(k[j]+k[m])/(2*h*h); inside=(k[j]+k[b])/(2*h*h)
  if kind=='periodic':
   out+=max(v[j] if j==0 else -v[j],0)/h
   inside+=max(-v[j] if j==0 else v[j],0)/h
  A[j,m]=-out; A[j,b]=-inside; A[j,j]=out+inside
 return A.tocsc()

def test(a,b):
 L=(kron(eye(ny),a)+kron(b,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.concatenate([splu(L+mu*I).solve(B)[rows] for mu in s['mu_shifts']])

kinds=['open','reflect','mirror','absorb','periodic_endpoint','periodic_diffusion','periodic']; ops={a:{m:op(a,m) for m in kinds} for a in 'xy'}
runs=[]
for x,y in product(kinds,repeat=2):
 r=test(ops['x'][x],ops['y'][y])-Y
 e=np.sqrt(np.mean(r*r,axis=0)); runs.append((max(e),x,y,e))
for q in sorted(runs)[:14]: print(q)
print('sweep fit check')
sw=s['sensor_calibration_sweeps']; print({k:np.shape(v) for k,v in sw.items() if k!='single_bank_sweeps'}); print({k:np.shape(v) for k,v in sw['single_bank_sweeps'].items()})
a=sw['raw_rosette']; b=sw['ref_rosette']; keep=np.all(abs(a)<=8.5,axis=1)
for j in range(4):
 c=np.linalg.lstsq(np.c_[np.ones(sum(keep)),b[keep,j]],a[keep,j],rcond=None)[0]; print(j,c,np.std(a[keep,j]-c[0]-c[1]*b[keep,j]))
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 t=np.asarray(e['facility_telemetry_trace']);print(e['completed_campaign_short_name'],t.shape,t.mean(axis=0).round(3),t.std(axis=0).round(3))
