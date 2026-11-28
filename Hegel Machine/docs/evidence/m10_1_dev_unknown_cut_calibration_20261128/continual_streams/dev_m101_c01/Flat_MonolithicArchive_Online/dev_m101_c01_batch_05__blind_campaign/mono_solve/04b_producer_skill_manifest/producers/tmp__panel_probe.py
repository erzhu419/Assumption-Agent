from probe import *
from scipy.linalg import svd
c=s['panel_c']; w=c['u']; np.set_printoptions(precision=5,suppress=True)

def basis(w,k,p):
 if k=='sq':return np.c_[aff(w),w*w]
 if k=='sqsin':return np.c_[aff(w),w*w,np.sin(w*p)]
 if k=='sqcos':return np.c_[aff(w),w*w,np.cos(w*p)]
 if k=='sqcubic':return np.c_[aff(w),w*w,w**3]
 if k=='sin':return np.c_[aff(w),np.sin(w*p)]
 if k=='sincos':return np.c_[aff(w),np.sin(w*p),np.cos(w*p)]
 if k=='tanh':return np.c_[aff(w),np.tanh(w*p)]
 if k=='tanhsq':return np.c_[aff(w),np.tanh(w*p),w*w]
 if k=='tanhsq2':return np.c_[aff(w),np.tanh(w*p),np.tanh(w*p)**2]
 if k=='rational':return np.c_[aff(w),w*w/(1+p*w*w),w**3/(1+p*w*w)]
 if k=='sinshift':return np.c_[aff(w),np.sin(w*p[:1]+p[1:])]
 if k=='tanhshift':return np.c_[aff(w),np.tanh(w*p[:1]+p[1:])]
 if k=='exp':return np.c_[aff(w),np.expm1(w*p)]

def panel(k,tar,mask):
 def err(p):
  f=basis(w[mask],k,p);return (f@fit(f,tar[mask])-tar[mask]).ravel()
 if k in ('sq','sqcubic'): p=np.array([1.])
 else:
  p0=([1.4]+[.1]*4) if k.endswith('shift') else [1.4]
  p=least_squares(err,p0,bounds=([.06]+[-3]* (len(p0)-1),[6]+[3]*(len(p0)-1)),max_nfev=100).x
 return p,fit(basis(w[mask],k,p),tar[mask])
ks=['sq','sqsin','sqcos','sqcubic','sin','sincos','tanh','tanhsq','tanhsq2','rational','sinshift','tanhshift','exp']; best={}
for tag in ('v1','v2'):
 tar=c[tag]; rank=[]
 for k in ks:
  es=[]
  for i in range(24):
   m=np.arange(24)!=i;p,bb=panel(k,tar,m);es.append(tar[i]-basis(w[i:i+1],k,p)[0]@bb)
  p,bb=panel(k,tar,np.ones(24,bool)); e=np.sqrt(np.mean(np.array(es)**2)); rank.append((e,k,p,bb));print(tag,k,'LOO',round(e,6),'p',p,'fit',np.std(tar-basis(w,k,p)@bb))
 rank.sort(key=lambda row:row[0]);best[tag]=rank[:3]
 for e,k,p,bb in rank[:2]: print('matrix',tag,k,bb)

splits=[np.arange(56,72),*np.array_split(np.random.default_rng(18).permutation(72),4)]
def check(f,tar=np.c_[y,z],ridge=0):
 errors=[]
 for ids in splits:
  m=np.ones(len(y),bool);m[ids]=0
  coef=fit(f[m],tar[m]) if not ridge else np.linalg.solve(f[m].T@f[m]+ridge*np.eye(f.shape[1]),f[m].T@tar[m]);e=np.mean(abs(tar[ids]-f[ids]@coef),0);errors.append([e[0],np.mean(e[1:])]);
 return np.r_[errors[0],np.max(errors,0)]
for name,u in [('x',x),('q',q),('r',r),('rx',aff(x)@B),('rq0',q@B[1:]),('rev',(q-B[0])@np.linalg.inv(B[1:]))]:
 p2=poly(u,2); print(name,'add34',check(np.c_[p2,u**3,u**4])); print(name,'harmonic',[(k,check(np.c_[p2,np.sin(k*u),np.cos(k*u)])) for k in (1.1,2,3)])
 for tag in ('v1','v2'):
  for _,k,p,bb in best[tag][:2]:
   op=basis(u,k,p)@bb; print(name,tag,k,'linear',check(aff(op)),'add',check(np.c_[aff(u),op]),'quadratic+op',check(np.c_[p2,op]),'op2',check(poly(op,2)))
 _,k,p,bb=best['v1'][0];g=basis(u,k,p)@bb;_,k,p,bb=best['v2'][0];h=basis(u,k,p)@bb;print(name,'joint ops',check(np.c_[aff(u),g,h]),'joint cross',check(np.c_[aff(u),g,h,g*h]),'compose',check(np.c_[aff(u),g,basis(g,k,p)@bb]))
