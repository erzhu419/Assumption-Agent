from discover import *
from scipy.optimize import least_squares
p=S['panel_b']; u,v=p['u'],p['v']; z=coords(S,S['train']['x']); tr=S['train']; T=tr['traj']; t=S['eval_times'];
models={
'cubic-rat2':lambda x,q:x**3/(1+q[0]*x*x),'cubic-rat4':lambda x,q:x**3/(1+q[0]*x**4),'cubic-ratpow':lambda x,q:x**3/(1+q[0]*x*x)**q[1], 'cubic-exp':lambda x,q:x**3*np.exp(-q[0]*x*x), 'cubic-log':lambda x,q:x*np.log1p(q[0]*x*x), 'cubic-clipped':lambda x,q:np.tanh(q[0]*x**3), 'asym-rat':lambda x,q:(x**3+q[1]*x*x)/(1+q[0]*x*x), 'shift-tanh':lambda x,q:np.tanh(q[0]*x+q[1]), 'shift-erf':lambda x,q:erf(q[0]*x+q[1]), 'shift-sin':lambda x,q:np.sin(q[0]*x+q[1]), 'shift-softplus':lambda x,q:np.logaddexp(0,q[0]*x+q[1]), 'cubic-mix':lambda x,q:(x**3+q[1]*x**5)/(1+q[0]*x*x), 'rat-mix':lambda x,q:(x**3+q[1]*x**5)/(1+q[0]*x*x+q[2]*x**4), 'cubic-aff':lambda x,q:x**3+q[0]*x**5+q[1]*x**7}
R=[]
for name,f in models.items():
 n=3 if name=='rat-mix' else 2 if name in ['cubic-ratpow','asym-rat','shift-tanh','shift-erf','shift-sin','shift-softplus','cubic-mix','cubic-aff'] else 1
 for extras in [0,1,2]:
  def G(x,q):return D(np.c_[x,f(x,q),x*np.roll(x,-1,axis=1)] if extras==1 else np.c_[x,f(x,q),x*x] if extras==2 else np.c_[x,f(x,q)])
  def res(q):
   F=G(u,q);return (F@fit(F,v)-v).ravel()
  opts=[least_squares(res,np.r_[a,np.full(n-1,b)],bounds=(np.r_[.00001,np.full(n-1,-5.)],np.r_[10.,np.full(n-1,10.)]),max_nfev=110) for a in [.22,1.25,2.9] for b in ([0.,1.] if n>1 else [0.])];o=min(opts,key=lambda o:o.cost); F=G(u,o.x);W=np.linalg.pinv(F); h=np.sum(F*W.T,1); loo=np.mean(abs(o.fun.reshape(v.shape)/(1-h[:,None])));R.append((loo,np.std(o.fun),name,extras,o.x, G(z,o.x)@fit(F,v)))
for r in sorted(R,key=lambda r:r[0])[:18]:print('B',r[:-1]);
b=min(R,key=lambda r:r[0])[-1];
# Profile downstream bases on four fixed subsamples without using their responses to estimate each model.
from sklearn.model_selection import KFold
splits=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=921).split(z)); C=bases(S,z); c=C['c20']; c1=C['c1']; A=np.c_[z,b]; time=np.c_[-np.expm1(-1.7936*t),-np.expm1(-.4178*t)]; Y=np.c_[tr['y'],T@np.linalg.pinv(time).T];
Gs={'z':z,'b':b,'c':c,'c1':c1,'raw':A,'z times b':(z[:,:,None]*b[:,None,:]).reshape(-1,16),'z roll pairs':z*np.roll(z,-1,axis=1),'b roll pairs':b*np.roll(b,-1,axis=1),'z cubics':z**3,'b cubics':b**3};
for k in (.5,1.25,2.5):
 for n,x in [('z',z),('b',b),('z+b',z+b),('z-b',z-b)]:
  Gs[str(k)+n+' tanh']=np.tanh(k*x);Gs[str(k)+n+' sin']=np.sin(k*x)
for target in range(3):
 sel=['raw']; print('TARGET',target)
 for it in range(5):
  cand=[]
  for name,f in Gs.items():
   X=D(np.c_[*[Gs[k] for k in sel],f]);
   if X.shape[1]>46 or name in sel:continue
   errs=[np.mean(abs(X[j]@fit(X[i],Y[i,target])-Y[j,target])) for i,j in splits];cand.append((np.mean(errs),errs[0],name))
  if not cand:break
  row=min(cand);sel.append(row[-1]);print(row)
 print('y/head residuals',np.round((Y-D(A)@fit(D(A),Y))[:12,target],3))
