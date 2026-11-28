from probe import *
from scipy.linalg import svd
from scipy.optimize import minimize_scalar
from itertools import product

def loo(F,Y):
 Q=np.linalg.qr(F,mode='reduced')[0]; e=(Y-Q@(Q.T@Y))/(1-np.sum(Q**2,axis=1))[:,None]; return np.mean(abs(e))
def basis(x,fs):return np.column_stack([np.ones(len(x))]+[f(x) for f in fs])
identity=lambda x:x
square=lambda x:x*x
cube=lambda x:x**3
funcs={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'atan':np.arctan,'softsat':lambda x:x/np.sqrt(1+x*x),'log1p':lambda x:np.sign(x)*np.log1p(abs(x))}
p=s['panel_c']; u=p['u']; specs={}
for key in ['v1','v2']:
 Y=p[key]; print('panel_c',key); vals=[]
 for tag,f in funcs.items():
  for fixed in [[],[identity],[square],[identity,square]]:
   def objective(a): return loo(basis(u,fixed+[lambda x:f(a*x)]),Y)
   sol=minimize_scalar(objective,bounds=(.12,2.8),method='bounded'); a=sol.x; fs=fixed+[lambda x,f=f,a=a:f(a*x)]
   vals.append((sol.fun,tag,len(fixed),a,fs))
 for fs in [[identity,square,cube],[identity,square,cube,lambda x:x**4],[identity,square]]: vals.append((loo(basis(u,fs),Y),'poly',len(fs),0,fs))
 vals.sort(key=lambda a:a[0]); print([v[:4] for v in vals[:10]]); specs[key]=vals[0][-1]
 for v in vals[:3]:
  F=basis(u,v[-1]); B=fit(F,Y); print(v[:4],np.linalg.svd(B[-4:],compute_uv=False),'rmse',np.std(F@B-Y)); print(B)

a=s['panel_a']; ia=np.arange(48); ia=ia[~np.isin(ia,[5,17,31])]; AA=np.array([fit(poly(a['u'][ia,i:i+1],1),a['v'][ia,i]) for i in range(4)]).T; b=s['panel_b']; B=fit(poly(b['u'],1),b['v']);
C={k:fit(basis(u,fs),p[k]) for k,fs in specs.items()}
ops={'a':lambda x:x*AA[1]+AA[0],'i':lambda x:(x-AA[0])/AA[1],'b':lambda x:poly(x,1)@B,'j':lambda x:(x-B[0])@np.linalg.inv(B[1:]),'c':lambda x:basis(x,specs['v1'])@C['v1'],'d':lambda x:basis(x,specs['v2'])@C['v2']}
x=s['train']['x']; z=s['train']['traj']; y=s['train']['y']; Y=np.c_[y,z]; tr=np.arange(56); te=np.arange(56,72)
rows=[]
transforms={'lin':[identity],'diag2':[identity,square],'diag3':[identity,square,cube],'cubic':None,'quad':None,'tanh':[np.tanh],'sin':[np.sin],'lin-tanh':[identity,np.tanh]}
for n in range(4):
 for names in product(ops,repeat=n):
  if len(set(names))<len(names) or any(set(names)>=set(v) for v in [('a','i'),('b','j')]):continue
  w=x
  for name in names:w=ops[name](w)
  for tag,fs in transforms.items():
   F=basis(w,fs) if fs else poly(w,2 if tag=='quad' else 3)
   pred=F[te]@fit(F[tr],Y[tr]); e=np.mean(abs(pred-Y[te]),axis=0); rows.append((e[0],e[1:].mean(),e[1:].max(),''.join(names) or 'raw',tag))
for j in (0,1):print('best target',j,*sorted(rows,key=lambda r:r[j])[:24],sep='\n')
