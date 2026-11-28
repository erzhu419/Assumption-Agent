from bases import *
from itertools import product
u=s['panel_c']['u']; h=s['panel_c']['v2']; q=s['panel_c']['v1']; pairs=[(0,1),(0,3),(1,2),(2,3)]
def qb(z):return np.c_[np.ones(len(z)),*[z[:,i]*z[:,j] for i,j in pairs]]
wq=ls(qb(u),q)
def Q(z):return qb(z)@wq
modes={'abs':lambda z,k:abs(z),'ratabs':lambda z,k:abs(z)/(1+k*abs(z)),'ratabssq':lambda z,k:abs(z)/(1+k*z*z),'ratsq':lambda z,k:z*z/(1+k*z*z),'ratlin':lambda z,k:z/(1+k*abs(z)),'softabs':lambda z,k:np.sqrt(z*z+k*k),'expabs':lambda z,k:-np.expm1(-k*abs(z)),'sinhabs':lambda z,k:np.sinh(k*abs(z)),'power':lambda z,k:abs(z)**k,'cos':lambda z,k:np.cos(k*z),'tanh':lambda z,k:np.tanh(k*z)}
res=[]
for n,fn in modes.items():
 for cross in [False,True]:
  for shift in [False,True]:
   def d(z,k):
    a=fn(z-(k[1:] if shift else 0),k[0]); return np.c_[poly(z,1),qb(z)[:,1:] if cross else np.empty((len(z),0)),a]
   def er(k):
    f=d(u,k); return (f@ls(f,h)-h).ravel()
   k0=[1.25]+([0.]*4 if shift else []); opt=least_squares(er,k0,bounds=([.0001]+([-1.]*4 if shift else []),[8.]+([1.]*4 if shift else [])),max_nfev=80)
   e=np.std(er(opt.x));res.append((e,n,cross,shift,opt.x))
print('C2 nonlinear hypotheses',*sorted(res,key=lambda v:v[0])[:15],sep='\n'); print('add abs coeffs',ls(np.c_[poly(u,1),abs(u)],h)); print('abs residual',h-np.c_[poly(u,1),abs(u)]@ls(np.c_[poly(u,1),abs(u)],h))
for a in [.22,1.25,1.]:
 for g in [u,abs(u),u*u]:
  for gg in [u,abs(u),u*u]:
   for label,f in [('rat',g/(1+a*gg)),('prod',g*(1+a*gg))]:
    f=np.c_[poly(u,1),f];e=np.std(f@ls(f,h)-h)
    if e<.07:print(a,label,e)
# Probe composition paths and sparse readouts. Rank on training only, also print untouched split.
def score(f,b):
 w=ls(f[:52],b[:52]); e=f@w-b
 return np.mean(abs(e[:52]),0),np.mean(abs(e[52:]),0)
base=[('raw',x),('A',x*sc+ca[0]),('Ai',(x-ca[0])/sc)]
allf=[]
for n,z in base:
 for n2,zz in [('I',z),('B',poly(z,1)@cb),('Bi',(z-cb[0])@np.linalg.inv(cb[1:]))]:
  for n3,a in [('I',zz),('Q',Q(zz)),('Q2',Q(Q(zz))),('QB',poly(Q(zz),1)@cb),('BQ',Q(poly(zz,1)@cb))]:allf.append((n+n2+n3,a))
for col,b in [('y',y[:,None]),('traj',t)]:
 r=[]
 for n,z in allf:
  for m,fn in [('lin',lambda z:poly(z,1)),('quad',lambda z:poly(z,2)),('abs',lambda z:np.c_[poly(z,1),abs(z)]),('sat',lambda z:np.c_[poly(z,1),z/(1+.22*z*z),z*z/(1+.22*z*z),np.tanh(1.25*z)])]:
   for j,f in [('only',fn(z)),('raw+',np.c_[fn(z),x]),('input+',np.c_[fn(z),dict(allf).get(n[:-2]+'I',x)])]:
    a,c=score(f,b);r.append((a.mean(),c.mean(),n,m,j))
 print(col,'composition candidates',*sorted(r)[:14],sep='\n')
