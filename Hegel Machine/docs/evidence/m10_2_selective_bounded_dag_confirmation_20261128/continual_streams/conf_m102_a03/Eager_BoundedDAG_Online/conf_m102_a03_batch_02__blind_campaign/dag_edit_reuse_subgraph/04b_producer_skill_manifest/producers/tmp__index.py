exec(open('/tmp/probe.py').read().split('for k,v in')[0])
from scipy.optimize import least_squares
p=s['panel_b']; b=fit(a(p['u']),p['v']); u=np.linalg.solve(b[1:].T,(z-b[0]).T).T
B=z[:,0]*z[:,2]+z[:,1]*z[:,3]; L=np.c_[a(z),B]; y=s['train']['y']; t=np.arange(len(y)); order=np.random.default_rng(42).permutation(len(y)); sp=[(t[:52],t[52:])]+[(np.setdiff1d(order,J),J) for J in np.array_split(order,6)];
func={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rational':lambda v:v/(1+v*v),'log cosh':lambda v:np.logaddexp(v,-v),'softsign':lambda v:v/np.sqrt(1+v*v),'lorentz':lambda v:1/(1+v*v),'gauss':lambda v:np.exp(-v*v),'cubic':lambda v:v**3,'mod':lambda v:1/(1+v*v)}
starts=[[0,0,0,q,c] for q in (0.75,1.5,3.) for c in (-.5,0,.5)]; starts+=list(np.c_[np.eye(4),np.zeros(4)]); starts=np.array(starts)
def get(i,k,mode):
 f=func[k]; Q=np.linalg.qr(L[i])[0]; w=y[i]-Q@(Q.T@y[i]);
 def extra(h,x):
  v=f(a(u[x])@np.r_[h[-1],h[:-1]]); return v[:,None] if mode==0 else (v[:,None]*B[x,None] if mode==1 else v[:,None]*a(u[x]))
 def obj(h):
  F=extra(h,i); F-=Q@(Q.T@F); n=np.linalg.norm(F,axis=0); F/=np.maximum(n,1e-13); return F@fit(F,w)-w
 opts=[least_squares(obj,h,bounds=(-5,5),max_nfev=130) for h in starts]; r=min(opts,key=lambda v:np.linalg.norm(v.fun));
 D=np.c_[L,extra(r.x,t)]; c=fit(D[i],y[i]); return D@c,r.x,c,np.mean(abs(r.fun))
for k in func:
 if k=='mod': continue
 for mode in ([0,1,2] if k in ('lorentz','gauss','tanh','sin','rational') else [0]):
  pred,h,c,e=get(t,k,mode); print(k,mode,round(e,7),np.round(h,4),np.round(c,4),flush=True)
  if e<.018:
   vals=[np.mean(abs(get(i,k,mode)[0][j]-y[j])) for i,j in sp]; print('CV',np.round([vals[0],np.mean(vals[1:]),max(vals[1:])],6),flush=True)
