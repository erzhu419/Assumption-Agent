exec(open('/tmp/opscan.py').read().split("print('base'")[0])
from itertools import combinations
u,v=(s['panel_b'][k] for k in ('u','v')); g=np.c_[y,a]; tele=s['telemetry_vec'];print('rates',dat['rates'],'tele',tele)
fun={'tanh':lambda q,p:np.tanh(p*q),'sin':lambda q,p:np.sin(p*q),'atan':lambda q,p:np.arctan(p*q),'erf':lambda q,p:__import__('scipy').special.erf(p*q),'alg':lambda q,p:q/np.sqrt(1+(p*q)**2),'rat':lambda q,p:q/(1+(p*q)**2),'gauss':lambda q,p:q*np.exp(-(p*q)**2),'cub':lambda q,p:q**3,'sq':lambda q,p:q*q,'smooth':lambda q,p:q/(1+p*abs(q))}
res=[]
for names in [(n,) for n in fun]+list(combinations(fun,2)):
 def f(q,p):return np.c_[aff(q),*[fun[n](q,b) for n,b in zip(names,p)]]
 def r(p):
  F=f(u,p);return (F@ls(F,v)-v).ravel()
 fits=[least_squares(r,[c]*len(names),bounds=(.08,7.),max_nfev=100) for c in (1.,2.15,3.)]; opt=min(fits,key=lambda k:k.cost); p=opt.x; w=ls(f(u,p),v); h=f(x,p)@w
 res.append((np.std(r(p)),names,p,cv(np.c_[aff(x),h]),w,h))
for R in sorted(res,key=lambda z:z[0])[:14]:print(R[:4])
for R in sorted(res,key=lambda z:z[3][1])[:5]:print('cv',R[:4])
for R in sorted(res,key=lambda z:z[0])[:3]:print(R[1],R[4]);
# Separate operator and readout contributions on identical 52/20 split.
for R in sorted(res,key=lambda z:z[0])[:5]:
 h=R[-1];print('readout',R[1], 'h',cv(aff(h)), 'hquad',cv(np.c_[aff(h),quad(h)]), 'hxquad',cv(np.c_[aff(x),h,quad(x)])); w=ls(np.c_[aff(x),h][:52],g[:52]); print('coef',np.round(w,3))
