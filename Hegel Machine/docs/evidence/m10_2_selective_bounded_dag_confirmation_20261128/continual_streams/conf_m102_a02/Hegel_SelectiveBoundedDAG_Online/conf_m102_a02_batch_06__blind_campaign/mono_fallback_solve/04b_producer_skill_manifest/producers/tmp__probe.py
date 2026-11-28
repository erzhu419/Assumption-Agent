import numpy as np,pickle,itertools,importlib.util
from scipy.linalg import lstsq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; z=tr['traj']; t=s['eval_times']; x=tr['x']
sp=importlib.util.spec_from_file_location('m','/tmp/hegel_continual_archive/conf_m102_a02_batch_03_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
a=m._fit_calibration(s['panel_a']); q=m._invert(x,a)
np.set_printoptions(precision=5,suppress=True); print('A',[(term,a[i]) for i,term in enumerate(m._TERMS) if np.any(a[i])]); print('q range',q.min(0),q.max(0)); print('A check',np.max(abs(m._design(q,35)@a-x)))
pa=s['panel_a']; print('A residual row norms',np.sort(np.linalg.norm(m._design(pa['u'],35)@a-pa['v'],axis=1)))
u,sv,vt=np.linalg.svd(z,full_matrices=False); print('sv',sv,'modes',vt[:3]); print('target correlations',np.corrcoef(np.column_stack([q,y,u[:,:3]]).T)[4:,:]);
def ls(f,v):return lstsq(f,v)[0]
def poly(f,d):return PolynomialFeatures(d).fit_transform(f)
def chk(f,Y=np.c_[y,z]):
 r=f[52:]@ls(f[:52],Y[:52])-Y[52:];return np.mean(abs(r[:,0])),np.mean(abs(r[:,1:]))
for nm,w in [('x',x),('q',q)]:
 for d in range(1,5):print(nm,d,chk(poly(w,d)))
for pn,vkey in [('b','v'),('c','v1'),('c','v2')]:
 p=s['panel_'+pn]; U=p['u']; V=p[vkey]; print('\n',pn,vkey)
 funcs={'square':lambda a:a*a,'cube':lambda a:a**3,'abs':np.abs,'signed-square':lambda a:a*abs(a),'sine':lambda a:np.sin(1.25*a),'cosine':lambda a:np.cos(1.25*a),'tanh':lambda a:np.tanh(1.25*a),'rational':lambda a:a/(1+1.25*a*a),'softabs':lambda a:np.sqrt(1+1.25*a*a),'exp':lambda a:np.exp(-1.25*a*a),'relu':lambda a:np.maximum(a,0)}
 bases={'aff':lambda w:np.c_[np.ones(len(w)),w], 'aff2':lambda w:np.c_[np.ones(len(w)),w,w*w]}
 res=[]
 for name,f in bases.items():
  for k,h in list(funcs.items())+[('none',lambda w:w[:,:0])]:
   F=np.c_[f(U),h(U)]; B=ls(F,V); r=F@B-V; L=np.einsum('ij,ji->i',F,np.linalg.pinv(F)); loo=r/np.maximum(1-L,1e-9)[:,None]; W=np.c_[f(q),h(q)]@B; res.append((np.mean(loo**2)**.5,np.mean(r**2)**.5,name,k,chk(poly(W,1)),chk(poly(W,2))))
 for row in sorted(res)[:9]:print(row)
