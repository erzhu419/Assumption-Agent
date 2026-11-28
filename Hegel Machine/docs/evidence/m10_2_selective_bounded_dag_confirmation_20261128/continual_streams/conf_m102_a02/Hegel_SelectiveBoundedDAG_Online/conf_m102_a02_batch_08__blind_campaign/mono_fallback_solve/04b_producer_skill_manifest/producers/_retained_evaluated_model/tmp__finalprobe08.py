exec(open('/tmp/expand08.py').read().split('R=[]')[0]);
from itertools import product
R=[]
for cyc, num, odd in product(['sum','each','shrink'],['fixed','fixed32','shared','individual','freeeven'],['none','h1','h3','both','scaled3','shiftcube']):
 def F(p):
  a=.22 if num=='fixed' else .32 if num=='fixed32' else p[:4] if num=='individual' else p[0]; h2=h[:,None]**2; inv=1/(1+a*h2)
  columns=[N,b*inv,C if cyc=='sum' else c[:,:] if cyc=='each' else np.c_[C,(c[:,[0,1,2]]-c[:,3,None])]]
  if num=='freeeven':columns.append(b*h2*inv)
  if odd in ['h1','both']:columns.append(b[:,3]*h)
  if odd in ['h3','both']:columns.append(b[:,3]*h**3)
  if odd=='scaled3':columns.append(b[:,3]*h**3*inv[:,0])
  if odd=='shiftcube':columns.append(b[:,3]*(h+p[-1])**3)
  return np.column_stack(columns)
 if cyc=='shrink':continue
 p0=[.22]*(4 if num=='individual' else 0 if num.startswith('fixed') else 1);bd=([0]*len(p0),[1.5]*len(p0))
 if odd=='shiftcube':p0.append(0);bd[0].append(-.5);bd[1].append(.5)
 R.append(((cyc,num,odd),run(F,p0,bd,.003)))
F=np.c_[full,b*h[:,None]**2,b[:,3]*h,b[:,3]*h**3];R.append(('compact_poly',run(lambda p:F)));R.append(('full_poly',run(lambda p:np.c_[full,b[:,[0,2,3]]*h[:,None],b*h[:,None]**2,b*h[:,None]**3])))
for k,v in sorted(R,key=lambda a:a[1][0])[:22]:print(k,fmt(v[:4]),v[4],v[5])
print('state symmetric CV',fmt(cv(base,Y=z)));print('full state CV',fmt(cv(full,Y=z)))
