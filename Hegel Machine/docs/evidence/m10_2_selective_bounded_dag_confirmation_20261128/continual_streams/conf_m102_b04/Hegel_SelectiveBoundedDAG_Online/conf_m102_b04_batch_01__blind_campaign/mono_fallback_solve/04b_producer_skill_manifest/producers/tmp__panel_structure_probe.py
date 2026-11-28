from operators import *
from scipy.optimize import least_squares
from itertools import product
p=s['panel_c']; u=p['u']; v=p['v1']; w=p['v2']; b=fit(basis(s['panel_b']['u'],1),s['panel_b']['v']); c=np.load('/tmp/cal.npy')
F={'atan':lambda x,a:np.arctan(a*x)/a,'tanh':lambda x,a:np.tanh(a*x)/a,'sin':lambda x,a:np.sin(a*x)/a,'sqrt':lambda x,a:x/np.sqrt(1+a*x*x),'rat':lambda x,a:x/(1+a*x*x),'soft':lambda x,a:x/(1+a*abs(x)), 'asinh':lambda x,a:np.arcsinh(a*x)/a, 'linear':lambda x,a:x}; E={'sq':lambda x,a:x*x,'ratsq':lambda x,a:x*x/(1+a*x*x),'normsq':lambda x,a:x*x/(1+a*np.sum(x*x,axis=1,keepdims=True)),'none':lambda x,a:x[:,:0]}
def panel(u,y,f,g,mode):
 def D(x,t):
  a=np.exp(t[:4]) if mode==4 else np.exp(t[0]); z=x+(t[mode:mode+4] if len(t)>mode+1 else 0); return np.c_[np.ones(len(x)),F[f](z,a),E[g](z,np.exp(t[-1]))]
 def fun(t):
  a=D(u,t);return (a@fit(a,y)-y).ravel()
 t0=np.r_[np.zeros(mode),np.zeros(4) if mode==4 else [],np.log(.22)]; r=least_squares(fun,t0,bounds=(np.r_[[-5]*mode,[-.6]*(len(t0)-mode-1),-8],np.r_[[2]*mode,[.6]*(len(t0)-mode-1),3]),max_nfev=100); a=D(u,r.x); co=fit(a,y); return np.std(loo(a,y)),np.std(r.fun),r.x,lambda x:D(x,r.x)@co
results={}
for name,Y,fs,gs in [('v',v,F,E),('w',w,['linear'],E)]:
 R=[]
 for f,g,m in product(fs,gs,[1,4]):
  if name=='w' and m==4:continue
  R.append((*panel(u,Y,f,g,m),f,g,m))
 R.sort(key=lambda v:v[0]);results[name]=R
 for looerr,err,t,fn,f,g,m in R[:8]:print(name,f,g,m,round(looerr,7),round(err,7),np.round(t,4))
x=s['train']['x'];z=s['train']['traj'];y=s['train']['y'];U,S,V=np.linalg.svd(z[:52],full_matrices=False);Y=np.c_[y,z@V[:1].T];
def cal(x):return (x-c[0])/c[1:].diagonal()
def rot(x):return basis(x,1)@b
D={}
for name,q in [('raw',x),('cal',cal(x)),('b',rot(x)),('bcal',rot(cal(x)))]:
 D[name]=q
 for k in ('v','w'):
  for j,R in enumerate(results[k][:2]):D[name+k+str(j)]=R[3](q)
for name,q in list(D.items())[:]:
 if len(name)<5 or name=='bcal':
  for f in ('sqrt','tanh','atan','rat'):
   for a in (.22,1.,1.25):D[name+f+str(a)]=F[f](q,a)
R=[]
for name,q in D.items():
 for d in (1,2,3):
  A=basis(q,d)
  for reg in (.00001,.001,.02):
   B=np.linalg.solve(A[:52].T@A[:52]+reg*np.eye(A.shape[1]),A[:52].T@Y[:52]); err=np.mean(abs(A@B-Y)[52:],0);R.append((err,name,d,reg))
for j in range(2):
 print('best basis',j)
 for r in sorted(R,key=lambda r:r[0][j])[:12]:print(r)
