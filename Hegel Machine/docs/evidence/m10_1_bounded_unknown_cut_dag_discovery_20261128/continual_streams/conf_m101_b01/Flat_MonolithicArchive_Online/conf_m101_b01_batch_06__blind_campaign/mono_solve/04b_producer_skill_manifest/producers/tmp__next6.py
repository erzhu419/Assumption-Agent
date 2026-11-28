from panel6 import *
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
import importlib.util
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m101_b01_batch_05_solver.py'); old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
x=old._inverse_calibration(tr['x'],old._calibration(a)); u,v=b['u'],b['v']
F={'exp':lambda u,k:u*(-np.expm1(-k*u*u)),'sin':lambda u,k:u-np.sin(k*u)/k,'rational':lambda u,k:u**3/(1+k*u*u),'tanh2':lambda u,k:u*np.tanh(k*u*u),'erf3':lambda u,k:erf(k*u**3),'alg2':lambda u,k:u**3/np.sqrt(1+k*u**4),'cos':lambda u,k:u*(1-np.cos(k*u)), 'tanh3':lambda u,k:np.tanh(k*u**3),'cube':lambda u,k:u**3}
ops={}
for name,f in F.items():
 def design(u,p):return af(f(u,p[0])+p[1]*u)
 def err(p):
  d=design(u,p);return (d@ls(d,v)-v).ravel()
 opt=min((least_squares(err,[k,j],bounds=([.02,-.6],[8,.8])) for k in (.25,1,3) for j in (0,.1)), key=lambda o:o.cost)
 print('B',name,opt.x,np.std(err(opt.x)))
 ops[name]=(lambda f,p,w:lambda z:af(f(z,p[0])+p[1]*z)@w)(f,opt.x,ls(design(u,opt.x),v))
z=old._operator_b(b,x); vh=ops['exp'](u); e=v-vh
print('B residual singulars',np.linalg.svd(e,compute_uv=False),'max rows',np.sort(np.linalg.norm(e,axis=1))[-8:]); print('B residual even terms coefficients',ls(np.c_[u*u,cross(u)],e).round(3))
z2=ops[min(ops,key=lambda k:np.mean((ops[k](u)-v)**2))](x)
d=lambda x:af(cross(x)); C1=lambda x:d(x)@ls(d(c['u']),c['v1']); d2=lambda x:np.c_[af(x),np.cos(1.28247*x)]; C2=lambda x:d2(x)@ls(d2(c['u']),c['v2']);
folds=[(np.setdiff1d(np.arange(52),i),i) for i in np.array_split(np.random.default_rng(624).permutation(52),4)]
final=(np.arange(52),np.arange(52,72)); results=[]
for name,X in [('x',x),('B',z),('newB',z2),('C1',C1(z2)),('C2',C2(z2)),('B+C',np.c_[z2,C1(z2),C2(z2)]),('x+B+C',np.c_[x,z2,C1(z2),C2(z2)]),('cube',x**3)]:
 for scale in (.7,1.5,3,6):
  for nug in (.0003,.01,.06):
   for power in (1,2,3):
    def fit(i,j):
     sd=np.maximum(X[i].std(0),1e-8); Xn=X/sd/np.sqrt(X.shape[1]); D=cdist(Xn,Xn)/scale
     if power==1: G=(1+np.sqrt(5)*D+5/3*D**2)*np.exp(-np.sqrt(5)*D)
     elif power==2: G=np.exp(-D**2/2)
     else: G=(1+D**2/2)**-1
     L=af((X-X[i].mean(0))/sd); G+= .3*L@L.T/L.shape[1];
     return abs(G[np.ix_(j,i)]@cho_solve(cho_factor(G[np.ix_(i,i)]+nug*np.eye(len(i))),Y[i])-Y[j]);
    E=np.concatenate([fit(i,j) for i,j in folds]); ef=fit(*final); results.append((name,scale,nug,power,E[:,0].mean(),E[:,1:].mean(),ef[:,0].mean(),ef[:,1:].mean()))
for idx in (4,5):
 print('GP rank',idx)
 for r in sorted(results,key=lambda r:r[idx])[:9]:print(r)
