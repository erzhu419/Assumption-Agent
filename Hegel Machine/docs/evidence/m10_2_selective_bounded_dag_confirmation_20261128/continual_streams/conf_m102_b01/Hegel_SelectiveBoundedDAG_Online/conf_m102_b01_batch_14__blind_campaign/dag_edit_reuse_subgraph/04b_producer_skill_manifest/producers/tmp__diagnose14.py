exec(open('/tmp/probe14.py').read().split("np.set_printoptions")[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=6,suppress=True)
for name,obj in s.items(): print(name, {k:np.shape(v) for k,v in obj.items()} if isinstance(obj,dict) else obj)
U=s['panel_a']['u']; V=s['panel_a']['v']; keep=np.all(abs(V)<3,axis=1); U=U[keep]; V=V[keep]; J=np.arange(4); K=(J+2)%4
for g in [2.4,2.45,2.5]:
 d=np.stack((np.ones_like(U),U,U[:,K],np.sin(g*U),np.sin(g*U[:,K])),axis=-1); q=np.stack([cal._ls(d[:,j],V[:,j]) for j in J]); r=V-(d*q).sum(-1); print('A',g,q,'noise',np.std(r,0)); print('maxabs',np.max(abs(r),0));
 if g==2.45:
  for name,f in [('square',U**2),('cube',U**3),('pair',U*U[:,K]),('square-cross',U**2*U[:,K]),('cross-square',U*U[:,K]**2),('adj_sine',np.sin(g*np.roll(U,1,axis=1))),('last_sine',np.sin(g*np.roll(U,-1,axis=1))),('joint_sine',np.sin(g*(U+U[:,K]))),('diff_sine',np.sin(g*(U-U[:,K])))]:
   rr=[]
   for j in J:
    q0=np.c_[d[:,j],f[:,j]]; Q=np.linalg.pinv(q0); e=V[:,j]-q0@(Q@V[:,j]); rr.append(np.sqrt(np.mean((e/(1-np.einsum('ij,ji->i',q0,Q)))**2)))
   print(name,rr)
forms={'asinh':np.arcsinh,'atan':np.arctan,'tanh':np.tanh,'inv3':lambda x:np.sinh(np.arcsinh(x)/3),'softsign':lambda x:x/np.sqrt(1+x*x),'erf': __import__('scipy.special',fromlist=['erf']).erf,'linear':lambda x:x,'sine':np.sin,'logsign':lambda x:np.sign(x)*np.log1p(abs(x))}
def feature(W,g,F,mode):
 Z=F(g*W); pair=[Z[:,i]*Z[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]; return np.column_stack((np.ones(len(W)),W,Z,*pair)) if mode==2 else (np.column_stack((np.ones(len(W)),Z,*pair)) if mode==1 else np.column_stack((np.ones(len(W)),W,Z)))
for name,F in forms.items():
 for mode,v in [(0,traj),(1,y),(2,y)]:
  stats=[]
  for a,b in folds:
   def loss(g):
    D=feature(w[a],g,F,mode); return np.mean((D@cal._ls(D,v[a])-v[a])**2)
   g=cal._profile(loss,.25,6); D=feature(w,g,F,mode); pred=D[b]@cal._ls(D[a],v[a]); stats.append((np.mean(abs(pred-v[b])),g,loss(g)**.5))
  stat=np.array(stats); print('head',name,mode, 'public',stat[0],'random mean',stat[1:].mean(0),'max',stat[1:,0].max())
print('common warps (public y/traj, worst-fold y/traj, average-fold y/traj)')
for q in [-.4,-.25,-.15,-.08,-.04,-.02,.02,.04,.08,.15,.25,.4]:
 for r in [0,.3,.6,1.]:
  W=w*(1+q/(1+(w/(r or .15))**2)) if r else w+q*w**3
  print(q,r,check(W))
