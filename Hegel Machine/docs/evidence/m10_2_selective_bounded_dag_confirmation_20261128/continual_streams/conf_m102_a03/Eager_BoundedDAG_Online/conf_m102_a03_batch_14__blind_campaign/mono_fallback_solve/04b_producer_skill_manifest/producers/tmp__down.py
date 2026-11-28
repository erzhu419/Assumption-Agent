from ops import *
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
F=features(x); Q=F['zB']; np.set_printoptions(precision=4,suppress=True)
qq=lambda z:np.column_stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
base=lambda z:aff(np.c_[z,qq(z)])
print('Q range',Q.min(0),Q.max(0))
print('y quadratic',ls(base(Q)[:52],y[:52])); print('traj quadratic',ls(base(Q)[:52],T[:52])[:,[0,4,7]])
folds=list(KFold(4,shuffle=True,random_state=131).split(y))
sp=[(np.arange(52),np.arange(52,72))]+folds
res=[]
def score(a,v):
 out=[]
 for i,j in sp:
  p=a[j]@ls(a[i],v[i]);out.append(np.mean(abs(p-v[j])))
 return out[0],np.mean(out[1:]),max(out[1:])
for target,v in [('y',y),('T',T)]:
 for n,z in F.items():
  if n not in ['z','x','zB','zBC','zBD','zC','zD']:continue
  for dn,a in [('lin',aff(z)),('sq',base(z))]:
   for gn, g in [('id',None)]+[(k+str(w),lambda z,g=g,w=w:g(z,w)) for k,g in fn.items() for w in [.22,.5,1,1.25,2.15,2.8,3.5,4.5]]:
    for dn2,b in [('one',a if g is None else np.c_[a,g(z)]),('two',None if g is None else np.c_[a,g(z),qq(g(z))])]:
     if b is None:continue
     res.append((target,*score(b,v),n,dn,gn,dn2))
 for metric in [1,2]: print('rank',target,metric,sorted([r for r in res if r[0]==target],key=lambda r:r[metric])[:12])
for gn,b in [('B',Q),('BC',F['zBC']),('BD',F['zBD'])]:
 for deg in [1,2]:
  a=aff(b) if deg==1 else base(b)
  for link in ['tanh','sin','asinh','linear']:
   out=[];pars=[]
   for i,j in sp:
    def apply(p,d):
     w=d@p[:-2];g={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'linear':lambda w:w}[link];return p[-2]*g(w)+p[-1]
    rs=[]
    for scale in [0.75,1.5,2.5,5.]:
     p0=np.r_[ls(a[i],y[i])/scale,scale,0.];r=least_squares(lambda p:apply(p,a[i])-y[i],p0,max_nfev=170);rs.append(r)
    r=min(rs,key=lambda r:r.cost);out.append(np.mean(abs(apply(r.x,a[j])-y[j])));pars.append(r.x)
   print('link',gn,deg,link, np.round(out,4),np.round(pars[0],3))
