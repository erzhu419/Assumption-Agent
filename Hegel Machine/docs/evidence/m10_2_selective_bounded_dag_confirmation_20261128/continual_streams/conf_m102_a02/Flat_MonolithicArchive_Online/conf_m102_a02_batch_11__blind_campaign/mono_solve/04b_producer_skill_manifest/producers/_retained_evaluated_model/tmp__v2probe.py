exec(open('/tmp/explore11.py').read().split("ops={'z':z}")[0])
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.optimize import least_squares
u=s['panel_c']['u']; v=s['panel_c']['v2']; zz=np.r_[u,z]; n0=len(u); rat=lambda x:x/(1+1.02*abs(x)**1.1225)
ii,jj=np.triu_indices(4); i,j=np.triu_indices(4,1)
print('u/v2',np.c_[u,v][:8].round(4)); print('v2 rms',np.std(v,axis=0));
def eval(f):
 F=f[:n0]; cv=[]
 for k in range(n0):
  keep=np.arange(n0)!=k; cv.append(F[k]@ls(F[keep],v[keep])-v[k])
 w=ls(F,v); return np.sqrt(np.mean((F@w-v)**2)),np.sqrt(np.mean(np.array(cv)**2)),f[n0:]@w,w

def ys(h):
 for nm,f in [('lin',aff(h)),('quad',quad(h)),('raw+op',np.c_[aff(z),h]),('rawq+op',np.c_[quad(z),h]),('raw+opq',np.c_[aff(z),quad(h)[:,1:]]),('raw+rat+op',np.c_[aff(z),rat(z),h])]:
  e=[]
  for sd in [-1,1,2,3,4,5]:
   idx=np.arange(72) if sd<0 else np.random.default_rng(sd).permutation(72); a,b=idx[:52],idx[52:]; e.append(np.mean(abs(f[b]@ls(f[a],y[a])-y[b])))
  print(nm,np.round([e[0],np.mean(e[1:]),max(e)],5),end='; ')
 print()
for nm,f in [('aff',aff(zz)),('quad',quad(zz)),('q+rat',np.c_[quad(zz),rat(zz)]),('q+tanh',np.c_[quad(zz),np.tanh(2.15*zz)]),('q+sin',np.c_[quad(zz),np.sin(2.15*zz)]),('q+cos',np.c_[quad(zz),np.cos(2.15*zz)]),('q+qrat',np.c_[aff(zz),rat(zz),rat(zz[:,ii])*rat(zz[:,jj])]),('q+sintele',np.c_[quad(zz),np.sin(2.15*zz),np.sin(1.25*zz)]),('q+pairrat',np.c_[quad(zz),rat(zz[:,i]+zz[:,j]),rat(zz[:,i]-zz[:,j])]),('q+paircos',np.c_[quad(zz),np.cos(2.15*(zz[:,i]+zz[:,j])),np.cos(2.15*(zz[:,i]-zz[:,j]))])]:
 er,ec,h,w=eval(f); print(nm,er,ec); ys(h)
 if nm in ('aff','quad','q+rat'): print(w.round(4))
r=v-quad(u)@ls(quad(u),v); print('quadratic panel residual',r.round(4))
