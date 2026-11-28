from panel_probe import s,fit,affine
import numpy as np,itertools
from scipy.optimize import minimize_scalar,least_squares
p=s['panel_a']; u=p['u'];v=p['v']; A=affine(u); keep=np.linalg.norm(v-A@fit(A,v),axis=1)<3
co=np.array([fit(A[keep][:,[0,i+1]],v[keep,i]) for i in range(4)]); z=(s['train']['x']-co[:,0])/co[:,1]
p=s['panel_c'];u=p['u'];v1=p['v1'];v2=p['v2']; B=affine(u)
opts={}
for k,v,fs in [(1,v1,{'rat':lambda x,a:x/(1+a*x*x),'tanh':lambda x,a:np.tanh(a*x),'atan':lambda x,a:np.arctan(a*x),'sin':lambda x,a:np.sin(a*x)}),(2,v2,{'log':lambda x,a:np.log1p(a*x*x),'exp':lambda x,a:np.expm1(-a*x*x),'cos':lambda x,a:np.cos(a*x)-1,'rat2':lambda x,a:x*x/(1+a*x*x)})]:
 for name,fn in fs.items():
  def design(x,a):return np.c_[affine(x),fn(x,a)]
  def err(a):
   D=design(u,a);return (D@fit(D,v)-v).ravel()
  q=least_squares(err,np.full(4,.6 if name not in ['sin','cos'] else 1.8),bounds=(.01,5),gtol=1e-12); scalar=minimize_scalar(lambda a:np.mean(err(a)**2),bounds=(.01,3),method='bounded'); print(k,name,scalar.x,scalar.fun**.5,q.x,np.std(q.fun))
  opts[name]=(fn,q.x,fit(design(u,q.x),v),fit(B,v))

def poly(x,degree):
 return np.column_stack([x[:,ids].prod(1) for d in range(1,degree+1) for ids in itertools.combinations_with_replacement(range(x.shape[1]),d)])
def cross(x):return np.c_[x, *[x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)]]
F={}
for name,a in [('raw',s['train']['x']),('cal',z),('b',affine(z)@fit(affine(s['panel_b']['u']),s['panel_b']['v']))]:
 for d in (2,3,4): F[f'{name}p{d}']=poly(a,d)
 F[name+'soft']=np.c_[poly(a,2),cross(np.tanh(a)),np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a)]
for name,(fn,a,w,b) in opts.items():
 T=fn(z,a); V=np.c_[affine(z),T]@w; R=(V-affine(z)@b)@np.linalg.pinv(b[1:]); N=T@w[5:]@np.linalg.pinv(w[1:5]);
 F[name+'basis']=cross(np.c_[z,T]); F[name+'vb']=np.c_[cross(V),poly(z,2)];
 for label,rr in [('R',R),('N',N),('T',T)]:
  for mult in [-1,.25,.5,1,2]:
   Z=z+mult*rr
   F[f'{name}{label}{mult}']=np.c_[poly(Z,2),cross(np.tanh(Z))]; F[f'{name}{label}{mult}p3']=poly(Z,3)
V1=np.c_[affine(z),opts['rat'][0](z,opts['rat'][1])]@opts['rat'][2]; V2=np.c_[affine(z),opts['log'][0](z,opts['log'][1])]@opts['log'][2]; F['all']=cross(np.c_[z,V1,V2]); F['both']=cross(np.c_[V1,V2]);F['exact_terms']=cross(np.c_[z,opts['rat'][0](z,opts['rat'][1]),opts['log'][0](z,opts['log'][1])]);
y=np.c_[s['train']['y'],s['train']['traj']]; n=len(z); print('n',n); splits=[(np.arange(n-20),np.arange(n-20,n))]+[(np.setdiff1d(np.arange(n),va),va) for va in np.array_split(np.random.default_rng(13).permutation(n),4)]
records=[]
for name,X in F.items():
 K=[]
 for tr,va in splits:
  x=(X-X[tr].mean(0))/np.maximum(1e-8,X[tr].std(0)); A=x[tr]@x[tr].T; ev,U=np.linalg.eigh(A); K.append((x[va]@x[tr].T@U,ev,U.T@(y[tr]-y[tr].mean(0)),y[tr].mean(0),y[va]))
 for reg in [.000001,.0001,.001,.01,.03,.1,.3,1,3]:
  E=[]
  for P,ev,B,m,target in K:
   err=abs((P/(ev+reg))@B+m-target); E.append([err[:,0].mean(),err[:,1:].mean()])
  rec=np.array(E); records.append((name,reg,rec[0],rec[1:].mean(0),rec[1:].max(0)))
for j in (0,1):
 print('target',j)
 for a in sorted(records,key=lambda x:.25*x[2][j]+.5*x[3][j]+.25*x[4][j])[:18]: print(a)
 print('public',sorted(records,key=lambda x:x[2][j])[:5])
