exec(open('/tmp/probe.py').read().split("print('cal',m)")[0]); z=(x-m[:,1])/m[:,0]
from scipy.optimize import least_squares
from itertools import combinations as cc
from scipy.linalg import block_diag
A=lambda u:np.c_[np.ones(len(u)),u]
ops={'sin':lambda u,k:np.sin(u*k),'tanh':lambda u,k:np.tanh(u*k),'rat':lambda u,k:u/(1+(k*u)**2),'cos':lambda u,k:np.cos(u*k),'sin3':lambda u,k:np.sin(k*u)**3,'cosdiff':lambda u,k:np.cos(k*u)-1,'sinu':lambda u,k:u*np.sin(k*u),'sin_cube':lambda u,k:np.sin(k*u**3),'atan':lambda u,k:np.arctan(k*u),'exp':lambda u,k:np.exp(-(k*u)**2)}
def fitop(u,V,name,base=A,shared=False):
 f=ops[name]; n=1 if shared else u.shape[1]; a=base(u)
 def d(k): return np.c_[a,f(u,k)]
 def fun(k):
  D=d(k); return (D@np.linalg.lstsq(D,V,rcond=None)[0]-V).ravel()
 out=min((least_squares(fun,np.full(n,k),bounds=(.1,8),max_nfev=150) for k in (.7,1.7,2.7,3.7,5.1)),key=lambda r:np.linalg.norm(r.fun)); k=out.x; B=np.linalg.lstsq(d(k),V,rcond=None)[0]; return np.sqrt(np.mean(out.fun**2)),k,lambda w:np.c_[base(w),f(w,k)]@B
P={}
for j,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[j]; R=sorted((fitop(p['u'],p[key],n,shared=True)[:2],n) for n in ops); print(j,key,R[:6]); n=R[0][1]; P[key]=fitop(p['u'],p[key],n,shared=True)[2]; print('independent',fitop(p['u'],p[key],n)[:2]);
F={k:f(z) for k,f in P.items()}; pair=lambda u:np.column_stack([u[:,i]*u[:,j] for i,j in cc(range(u.shape[1]),2)])
for j in (0,8):
 for n in ('sin','tanh','rat','atan'):
  for q in (False,True):
   base=(lambda u:feat(u,2)) if q else A
   print('target',j,n,q,fitop(z,Y[:,j],n,base,True)[:2])
B={'x':z, **F,'xp':pair(z),'bp':pair(F['v']),'cp':pair(F['v1']),'dp':pair(F['v2']),'bs':F['v']**2,'cs':F['v1']**2,'ds':F['v2']**2,'x2':z*z,'x4':z**4,'xv':z*F['v'],'vc':F['v']*F['v1'],'vd':F['v']*F['v2'],'cd':F['v1']*F['v2'],'tanh':np.tanh(2*z),'sin':np.sin(3*z)}
for j in (0,1):
 results=[]
 for n in range(1,5):
  for ks in cc(B,n):
   D=np.c_[np.ones(len(z)),*[B[k] for k in ks]]; e=cv(D,Y,.003)[j]; results.append((max(e),e[0],ks))
 print('best fixed designs',j,sorted(results)[:18]);
for k in F: print(k,cv(np.c_[A(z),F[k]],Y));
