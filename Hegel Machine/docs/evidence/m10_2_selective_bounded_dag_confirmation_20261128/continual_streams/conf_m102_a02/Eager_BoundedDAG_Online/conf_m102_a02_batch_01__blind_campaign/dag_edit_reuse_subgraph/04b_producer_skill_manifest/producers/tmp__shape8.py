exec(open('/tmp/grammar6.py').read().split('for pk,names')[0])
from scipy.special import erf
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True)
N=np.arange(len(x)); folds=[N[52:]]+list(np.array_split(np.random.default_rng(17).permutation(N),6))+[j for k in range(4) for j in (np.argsort(x[:,k])[:12],np.argsort(x[:,k])[-12:])]
warp={ 'rat3':lambda z,a:z**3/(1+a*z*z), 'tanh3':lambda z,a:np.tanh(a*z)**3, 'sin3':lambda z,a:np.sin(a*z)**3, 'tanhcub':lambda z,a:np.tanh(a*z**3), 'atan3':lambda z,a:np.arctan(a*z)**3, 'erf3':lambda z,a:erf(a*z)**3, 'power':lambda z,a:np.sign(z)*abs(z)**a, 'rational':lambda z,a:(.22*z+z**3)/(1+a*z*z), 'softclip':lambda z,a:np.tanh(a*(.22*z+z**3)), 'asinh':lambda z,a:np.arcsinh(a*(.22*z+z**3))}
def features2(z,w,option):
 if option==0:return np.c_[one(z),w]
 if option==1:return np.c_[one(z),w,z]
 if option==2:return np.c_[one(z),w,z, q(w,pairs),w*w]
 if option==3:return np.c_[one(z),w,z, q(z,pairs)]
 if option==4:return np.c_[one(z),w,z, q(w,pairs)]
scale=Y.std(axis=0)
for name,wf in warp.items():
 for opt in (1,2,3,4):
  def f(z,p):return features2(z,wf(z,p[0]),opt)
  def fit(z,y):
   def residual(p):
    A=f(z,p); return ((A@np.linalg.lstsq(A,y,rcond=None)[0]-y)/scale).ravel()
   res=least_squares(residual,[2.8 if name=='power' else .6],bounds=([.02],[4.]),max_nfev=30)
   A=f(z,res.x);return res.x,np.linalg.lstsq(A,y,rcond=None)[0],np.mean(abs(res.fun))
  P,W,R=fit(x,Y); es=[]; ps=[]
  for j in folds:
   i=np.setdiff1d(N,j); p,w,r=fit(x[i],Y[i]); e=abs(f(x[j],p)@w-Y[j]);es.append([e[:,0].mean(),e[:,1:].mean()]);ps.append(p)
  e=np.array(es); print(name,opt,'param',P,'fit',round(R,4),'official',e[0],'randommax',e[:7].max(0),'edges',e[7:].max(0))
for k,name in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 z,y=s[k]['u'],s[k][name]; print(k,name)
 mods={**warp,'atan':lambda z,a:np.arctan(a*z),'tanh':lambda z,a:np.tanh(a*z),'sin':lambda z,a:np.sin(a*z),'cubic':lambda z,a:z**3+a*z,'quadratic':lambda z,a:z*z+a*z,'erf':lambda z,a:erf(a*z)}
 for n,wf in mods.items():
  def resid(p):
   A=np.c_[one(z),wf(z,p[0])];return A@np.linalg.lstsq(A,y,rcond=None)[0]-y
  r=least_squares(lambda p:resid(p).ravel(),[.6],bounds=([.001],[6.]));print(n,r.x,np.mean(abs(r.fun)))
