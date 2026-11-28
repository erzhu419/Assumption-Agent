from wall_probe import *
ref,raw=[s['sensor_calibration_sweeps'][k] for k in ('ref_rosette','raw_rosette')]
print('upstream payload',pay)
for j in range(4):
 ok=np.abs(raw[:,j])<=8.5
 print('sweep',j,np.polyfit(ref[ok,j],raw[ok,j],1),'payload max err',np.max(np.abs(raw[ok,j]-ref[ok,j]*pay['gains'][j]-pay['offsets'][j])))

def ax(k,v,h,r,kind):
 a=axis(k,v,h,r,([1,1,1,1] if kind=='open' else [0,0,1,1])).tolil()
 if kind in ('periodic','wrap_add','wrap_local','wrap_half'):
  q=(k[0]+k[-1])/(2*h*h)
  for i,j,t in ((0,len(k)-1,1),(len(k)-1,0,-1)):
   p=i+t
   wrap=q+max(t*v[i],0)/h if kind=='periodic' else (q if kind!='wrap_local' else k[i]/h**2)
   a[i,j]=-wrap
   if kind=='periodic': a[i,p]=-(k[i]+k[p])/(2*h*h)-max(-t*v[i],0)/h
   if kind=='wrap_half': a[i,p]*=.5
   a[i,i]=-a[i,p]-a[i,j]
 if kind=='reflect':
  for i,j in ((0,1),(-1,-2)): a[i,j]*=2; a[i,i]*=2
 return a.tocsc()

from itertools import product
V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
kinds=['open','sealed','periodic','wrap_add','wrap_local','wrap_half','reflect']; pairs=[]
for x,y in product(kinds,repeat=2):
 X=ax(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],x);A=ax(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],y)
 L=sparse.kron(sparse.eye(ny),X,format='csc')+sparse.kron(A,sparse.eye(nx),format='csc')+V
 yy=np.vstack([splu(L+mu*I).solve(rhs)[at] for mu in s['mu_shifts']]); e=yy-Y
 pairs.append((np.max(np.sqrt(np.mean(e**2,axis=0))),x,y,np.sqrt(np.mean(e**2,axis=0)),np.max(abs(e))))
for a in sorted(pairs)[:12]: print(a)
print('aux ranges', np.min(s['aux_manifold_telemetry'],axis=0), np.max(s['aux_manifold_telemetry'],axis=0))
