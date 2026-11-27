exec(open('/tmp/chamber_probe.py').read().split('for k,x in')[0])
from itertools import product
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

def axis(k,v,h,r,typ,q=None):
 w=np.ones(4)
 if typ in ('seal','wrap','upwrap','mirror','cross'): w[:2]=0
 if typ=='mirror': w[2:]=2
 a=m._axis(k,v,h,r,w).tolil()
 if typ in ('wrap','upwrap','cross'):
  face=(k[0]+k[-1])/(2*h*h)
  for i,j,l in [(0,1,-1),(-1,-2,0)]:
   u=0 if typ!='upwrap' else (v[i] if i==0 else -v[i])/h
   kji=(k[i]+k[j])/(2*h*h) if typ=='upwrap' else k[i]/h**2
   a[i,j]=-kji-max(-u,0)
   a[i,l]=-face-max(u,0)
   a[i,i]=-a[i,j]-a[i,l]
  if typ=='cross':
   for i,j,l in [(0,1,-1),(-1,-2,0)]:
    a[i,j],a[i,l]=a[i,l],a[i,j]
 return a.tocsc()

def pred(ax,ay):
 nx,ny=s['nx'],s['ny']; ident=eye(nx*ny,format='csc'); op=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); rhs=modes.transpose(0,2,1).reshape(len(modes),-1).T; ix=ij[:,0]+nx*ij[:,1]; return np.vstack([splu(op+mu*ident).solve(rhs)[ix] for mu in s['mu_shifts']])
def make(d,typ):
 return axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1),s['robin_'+d],typ)
def score(a): return np.sqrt(np.mean((a-obs).reshape(3,-1,2)**2,axis=1))
types=['open','seal','mirror','wrap','upwrap','cross']; best=[]
for a,b in product(types,repeat=2):
 e=score(pred(make('x',a),make('y',b))); best.append((e.max(),a,b,e.tolist()))
for row in sorted(best)[:10]: print('discrete',row)

def flexible(d,q,kind):
 k,v=s['kappa_'+d],s['vel_'+d]; h=s['L'+d]/(s['n'+d]-1); r=s['robin_'+d]; a=m._axis(k,v,h,r,q[:4]).tolil()
 for i,j,l,t in [(0,1,-1,0),(-1,-2,0,1)]:
  gain=q[4+t]*k[i]/h**2; a[i,i]+=gain; a[i,l if kind=='wrap' else (2 if i==0 else -3)]-=gain
 return a.tocsc()
for kind in ['wrap','second']:
 for j in [-1,0,1]:
  idx=slice(None) if j<0 else j
  def p(q): return pred(flexible('x',q[:6],kind),flexible('y',q[6:],kind))
  init=([0,0,1,1,.5,.5] if kind=='wrap' else [0,0,1,1,0,0])*2
  fit=least_squares(lambda q:(p(q)-obs)[:,idx].ravel(),init,bounds=(-np.tile([.2,.2,.1,.1,.1,.1],2),10),max_nfev=140)
  print(kind,j,np.round(fit.x,5),np.round(score(p(fit.x)),7))
