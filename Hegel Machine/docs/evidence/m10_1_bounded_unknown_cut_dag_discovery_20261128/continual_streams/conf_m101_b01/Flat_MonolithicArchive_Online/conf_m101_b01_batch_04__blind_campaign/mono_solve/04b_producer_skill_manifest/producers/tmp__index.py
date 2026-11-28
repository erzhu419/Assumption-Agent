from shape import *
from scipy.optimize import minimize_scalar
P=s['panel_c']; u=P['u']; sp=lambda z:np.logaddexp(0,2.2*z)
r=aff(sp(b))@ls(aff(sp(u)),P['v2']); H=lambda z:z/(1+.47446*abs(z))**2
h=H(b); q=aff(h)@ls(aff(H(u)),P['v1']); print('y stats',np.min(y),np.max(y),flush=True)
scale=lambda z:(z-z.mean(0))/z.std(0)
Z={'b':scale(b),'q':scale(q),'r':scale(r),'h':scale(h)}
fs={'tanh':np.tanh,'sin':np.sin,'rat':lambda z:z/(1+z*z),'gauss':lambda z:np.exp(-z*z),'soft':lambda z:np.logaddexp(0,z),'abs':lambda z:np.abs(z)}
S=[]
def run(z,d,fn,i,j,starts=6):
 A=d[i]; reg=np.maximum(np.std(A,axis=0),.06)*.045; reg[0]=0
 B=np.r_[A,np.diag(reg)]; v=np.r_[y[i],np.zeros(len(reg))]; Q=np.linalg.qr(B,mode='reduced')[0]; vr=v-Q@(Q.T@v)
 def col(p):
  c=fn(z[i]@p[:-1]+p[-1]); c=np.r_[c,np.zeros(len(reg))]; return c-Q@(Q.T@c)
 def loss(p):
  c=col(p); return np.r_[vr-c*((c@vr)/(c@c+.0009)), .005*p]
 rng=np.random.default_rng(123); e=np.eye(z.shape[1]); init=[np.r_[1.2*e[k],0.] for k in range(len(e))]+[np.r_[rng.normal(0,.7,len(e)),rng.normal(0,.2)] for _ in range(starts)]
 fits=[least_squares(loss,p,bounds=(-5,5),max_nfev=55) for p in init]; f=min(fits,key=lambda q:q.cost); c=fn(z@f.x[:-1]+f.x[-1]); D=np.c_[d,c]; w=ls(np.r_[D[i],np.diag(np.r_[reg,.03])],np.r_[y[i],np.zeros(len(reg)+1)]); return np.mean(abs(D[j]@w-y[j])),np.mean(abs(D[i]@w-y[i])),f.x.round(2)
for k,z in Z.items():
 for base,d in [('linear',aff(z)),('quad',poly(z)),('shared',np.c_[aff(h),h[:,I]*h[:,J],b] if k in ('b','q') else np.c_[aff(z),h])]:
  for name,fn in fs.items():
   sc=[run(z,d,fn,i,j) for i,j in folds]; S.append((max(a[0] for a in sc),k,base,name,sc))
for v in sorted(S)[:20]:print(v,flush=True)
