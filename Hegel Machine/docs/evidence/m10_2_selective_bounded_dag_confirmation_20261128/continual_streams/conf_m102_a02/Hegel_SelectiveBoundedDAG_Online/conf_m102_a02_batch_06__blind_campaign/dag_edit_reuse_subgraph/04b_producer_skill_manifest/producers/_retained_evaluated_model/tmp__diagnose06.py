exec(open('/tmp/fit06.py').read().split("for name,h in [('cal',q)")[0])
from scipy.optimize import least_squares
U,V=b['u'],b['v']; print('B frozen rmse',np.sqrt(np.mean((panelb(U)-V)**2))); print('ranges',U.min(0),U.max(0)); print('C v1 entries',np.round(np.c_[c['u'],c['v1'],c['v2']][:10],3));
funcs={'rational':lambda x,p:x/(1+p[0]*x*x),'tanh':lambda x,p:np.tanh(p[0]*x),'arctan':lambda x,p:np.arctan(p[0]*x),'softsign2':lambda x,p:x/np.sqrt(1+p[0]*x*x),'sin':lambda x,p:np.sin(p[0]*x),'tanh sq':lambda x,p:np.tanh(p[0]*x*x),'sin sq':lambda x,p:np.sin(p[0]*x*x),'cubic':lambda x,p:x**3,'oddpoly':lambda x,p:np.c_[x**3,x**5],'abs':lambda x,p:abs(x),'exp':lambda x,p:x*np.exp(-p[0]*x*x)}
for target in ('B','C1','C2'):
 u,v=(U,V) if target=='B' else (c['u'],c['v'+target[-1]])
 for name,fn in funcs.items():
  for deg in (1,2):
   if len(u)<= (5 if deg==1 else 15)+4*(2 if name=='oddpoly' else 1): continue
   def feat(x,p):return np.c_[design(x,deg),fn(x,p)]
   def res(p):
    f=feat(u,p);return (f@ls(f,v)-v).ravel()
   p=min((least_squares(res,[p],bounds=(.001,20),max_nfev=80) for p in (.2,1,3,7)),key=lambda r:r.cost).x
   f=feat(u,p); w=ls(f,v); out=lambda x:feat(x,p)@w
   err=np.sqrt(np.mean(res(p)**2)); print(target,name,deg,p.round(4),round(err,6),end=' ')
   for h in (out(q),np.c_[q,out(q)]):
    ds=design(h,2 if h.shape[1]==4 else 1); print(round(np.mean(abs(ds[52:]@ls(ds[:52],tr[:52])-tr[52:])),5),end=' ')
   print()
