exec(open('/tmp/quot08.py').read().split('for key,v')[0])
from itertools import combinations
from scipy.optimize import minimize
cc=z*np.roll(w,1,axis=1); z4=z[:,3]; A=aff(w); zz=pol(z,2)[:,4:]; Q=pol(w,2)[:,4:]; W=cross(z,w); mask=np.ones((4,4),bool); mask[np.arange(4),(np.arange(4)-1)%4]=False
print('training linear+cyclic coefficients',np.linalg.lstsq(np.c_[A,cc]/(1+.2*z4[:,None]**2),y,rcond=None)[0]);print('telemetry',s['telemetry_vec'])
J=np.arange(n); sp += [(J[np.argsort(z[:,k])[:-12]],J[np.argsort(z[:,k])[-12:]]) for k in range(4)]
rows=[]
for key,core in [('cyclic',cc),('equalcyc',cc.sum(1,keepdims=True)),('full',W),('cyclicZ',np.c_[cc,z]),('equalZ',np.c_[cc.sum(1,keepdims=True),z]),('cyclicQ',np.c_[cc,Q])]:
 for loc in ['all','int','lin','separate']:
  for extra in ['', 'center','offset','bias','quadratic','correct']:
   def model(i):
    sc=np.ones(1)
    def fit(q,ret=False):
     k=q[0]; d=z4-(q[1] if extra=='center' else 0); fac=1+k*d*d; F=fac[:,None]; G=(1+q[-1]*d*d)[:,None] if loc=='separate' else F
     P=np.c_[A/F if loc in ('all','lin','separate') else A,core/G if loc in ('all','int','separate') else core]
     if extra=='offset':P=np.c_[P,np.ones(n)]
     if extra=='bias':P=np.c_[P,z4,z4/F[:,0]]
     if extra=='quadratic':P=np.c_[P,zz/F]
     if extra=='correct':P=np.c_[P,w*d[:,None]**2/F]
     c=np.linalg.lstsq(P[i],y[i],rcond=None)[0]; err=np.square(P[i]@c-y[i]).sum()
     return (P@c,c) if ret else err
    bounds=[(0,.8)]+([(-.6,.6)] if extra=='center' else [])+([(0,.8)] if loc=='separate' else []); r=minimize(fit,[.2]+([0] if extra=='center' else [])+([.2] if loc=='separate' else []),bounds=bounds,method='L-BFGS-B'); yh,c=fit(r.x,True); return yh,r.x,c
   out=[model(i) for i,j in sp]; ee=np.array([np.abs(o[0][j]-y[j]).mean() for o,(i,j) in zip(out,sp)]); ef=np.array([np.abs(o[0][i]-y[i]).mean() for o,(i,j) in zip(out,sp)]); rows.append((ee[:6].mean(),key,loc,extra,ee.round(4),ef[:6].mean(),out[0][1].round(3),out[0][2].round(3)))
print('best models');print(*sorted(rows)[:18],sep='\n'); print('least complexity');print(*[a for a in rows if a[1] in ('cyclic','equalcyc') and a[3]==''],sep='\n')
for name, D in [('h',aff(h)),('mixed',m._mixed(z,h)),('w-cyc',np.c_[aff(w),cc]),('w-cyc-z',np.c_[aff(w),cc,z])]:
 for penal in [.001,.015]:
  out=[]
  for i,j in sp:
   c=m._ridge(D[i],D[j],T[i],penal);out.append(np.abs(c-T[j]).mean())
  print('state',name,penal,np.round(out,5))
