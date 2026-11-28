import pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=130)
def poly(u,d):
 ids=[t for k in range(1,d+1) for t in itertools.combinations_with_replacement(range(4),k)]; return np.c_[np.ones(len(u)),*[u[:,t].prod(1) for t in ids]], [()]+ids
def lst(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
for pn in ('panel_a','panel_c'):
 p=s[pn];u=p['u']; X,_=poly(u,1)
 for vn,v in p.items():
  if vn=='u':continue
  m=np.ones(len(v),bool)
  if pn=='panel_a':
   for _ in range(6):
    r=np.linalg.norm(X@lst(X[m],v[m])-v,axis=1);m=r<5*np.median(r)
  a,b=u[m],v[m];x=X[m];print(pn,vn,'ranges',b.min(0),b.max(0));
  for name,q,ids in [('linear',*poly(a,1)),('quadratic',*poly(a,2)),('cubic',*poly(a,3)),('diag cubic',np.c_[x,a**2,a**3],[()]+[(i,)*j for j in (1,2,3) for i in range(4)])]:
   w=lst(q,b); r=q@w-b; H=np.sum(np.linalg.svd(q,full_matrices=False)[0]**2,axis=1);print(name,'rms',np.std(r,0),'LOO',np.sqrt(np.mean((r/np.maximum(1-H[:,None],1e-9))**2,0))); 
   if name=='cubic' and pn=='panel_a' or name=='quadratic' and pn=='panel_c':
    for j in range(4):print(j,[(ids[i],round(w[i,j],4)) for i in np.argsort(abs(w[:,j]))[-9:][::-1]])
  for fn in [np.sin,np.cos,np.tanh]:
   def obj(f):
    q=np.c_[x,fn(f*a)];return np.mean((q@lst(q,b)-b)**2)
   f=minimize_scalar(obj,bounds=(.1,5),method='bounded').x; print(fn.__name__,f,obj(f)**.5)
  for kind in ('tanh','sin','asinh'):
   j=0
   while j<4:
    y=b[:,j]; mid=(y.max()+y.min())/2; span=np.max(abs(y-mid)); fun={'tanh':np.arctanh,'sin':np.arcsin,'asinh':np.arcsinh}[kind]; inv={'tanh':np.tanh,'sin':np.sin,'asinh':np.sinh}[kind]
    def obj(sc):
     z=fun((y-mid)/sc);return np.std(inv(x@lst(x,z))*sc+mid-y)
    sc=minimize_scalar(obj,bounds=(span*1.00001,10*span),method='bounded').x; print(kind,j,round(sc,4),round(obj(sc),5),end='; ');j+=1
   print()
