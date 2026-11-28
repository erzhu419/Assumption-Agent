from operator04 import *
from operator04 import _affine, _fit, _products
from scipy.optimize import minimize_scalar
u,v=s['panel_b']['u'],s['panel_b']['v']
def rm(e):return np.sqrt(np.mean(e*e))
def profile(f,bounds):
 def ff(x,k): return f(x,k)
 def score(k):
  d=_affine(ff(u,k)); e=v-d@_fit(d,v); return rm(e/(1-(d*np.linalg.pinv(d).T).sum(1))[:,None])
 k=minimize_scalar(score,bounds=bounds,method='bounded').x
 d=_affine(ff(u,k)); w=_fit(d,v)
 return score(k),k,_affine(ff(Z,k))@w,w
models={}
for name,f,r in [('tanhsub',lambda x,k:x-np.tanh(k*x)/k,(.1,4)),('rat3',lambda x,k:x**3/(1+k*x*x),(.001,6)),('atanSub',lambda x,k:x-np.arctan(k*x)/k,(.1,4)),('rootSub',lambda x,k:x-x/np.sqrt(1+k*x*x),(.001,6)),('sinh',lambda x,k:np.sinh(k*x)-k*x,(.1,4)),('signedPower',lambda x,k:np.sign(x)*abs(x)**k,(1.1,5)),('logsub',lambda x,k:x*np.log1p(k*x*x),(.001,6))]:
 cv,k,b,w=profile(f,r); print('B structural',name,cv,k); models[name]=b
for key,f,k in [('Btanh','tanh',1.4),('Brat','rat',.87)]:models[key]=panel('panel_b','v',f,k,0)
c1=panel('panel_c','v1','atan',2.38,0); c2=panel('panel_c','v2','log',.602,0)
perm=np.random.default_rng(795).permutation(72); folds=[np.arange(52,72),np.arange(56,72),*np.array_split(perm,4),np.argsort(abs(Z[:72]).max(1))[-16:]]
def check(d):
 rows=[]
 for j in folds:
  i=np.setdiff1d(np.arange(72),j); e=abs(d[j]@_fit(d[i],Y[i])-Y[j]); rows.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
 return np.round(np.r_[rows[0],np.max(rows,axis=0)],5)
for name,b in models.items():
 for n,d in [('aff',np.c_[_affine(Z),b]),('bquad',np.c_[_affine(Z),b,_products(b)])]:print(name,n,check(d))
b=models['Brat']; bb=np.c_[_affine(Z),b]; dd=np.c_[bb,_products(b)]; w=_fit(dd[:72],Y); print('Bquad readout coefficients scalar/first/last:\n',w[:,[0,1,-1]]); print('Bquad residual sv',np.linalg.svd(Y-dd[:72]@w,compute_uv=False));
for name,d in [('zquad',np.c_[dd,_products(Z)]),('B*C2',np.c_[dd,b*c2]),('C1',np.c_[dd,c1]),('C2',np.c_[dd,c2]),('B*C1',np.c_[dd,b*c1]),('B*z',np.c_[dd,b*Z]),('C1(B)',np.c_[dd,np.arctan(2.3765*b)]),('log(B)',np.c_[dd,np.log1p((.602*b)**2)]),('B^3',np.c_[dd,b**3])]:print(name,check(d))
