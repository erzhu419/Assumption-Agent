from combine import *
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
p=s['panel_b'];u=p['u'];v=p['v']
# Compare shared univariate B links against leakage or interactions between the pairs.
def basis(x,kind,k):
 if kind=='rat':return x**3/(1+k*x*x)
 if kind=='rat5':return x**5/(1+k*x*x)
 if kind=='exp':return x*(-np.expm1(-k*x*x))
 if kind=='tanh':return np.tanh(k*x)**3
 if kind=='sin':return np.sin(k*x)**3
 if kind=='poly':return x**3
 if kind=='soft':return x**3/(1+k*abs(x))
 if kind=='pow':return np.sign(x)*abs(x)**k

def fitB(u,v,kind,k,mode):
 def fn(x,I):
  q=x[:,I]; a=basis(q,kind,k)
  fs=[q,a]
  if mode in ('power','all'):fs.append(a*q*q)
  if mode=='pair':fs.extend([(q[:,0]*q[:,1])[:,None], (q[:,0]*q[:,1])[:,None]*q])
  return bias(np.column_stack(fs))
 fits=[]
 for I,J in [([0,3],[0,1]),([1,2],[2,3])]:
  I=list(range(4)) if mode in ('wide','all') else I
  F=fn(u,I);C=ls(F,v[:,J]);fits.append((I,C))
 return lambda x:np.column_stack([fn(x,I)@C for I,C in fits])

def mapB(u,v,kind,mode):
 def obj(k):return np.mean((fitB(u,v,kind,k,mode)(u)-v)**2)
 opt=minimize_scalar(obj,bounds=(1.1,7) if kind=='pow' else (.03,3),method='bounded')
 return fitB(u,v,kind,opt.x,mode),opt.x,np.sqrt(opt.fun)

if __name__=='__main__':
 rng=np.random.default_rng(12);results=[]
 for kind in ['rat','rat5','exp','tanh','sin','poly','soft','pow']:
  for mode in ['strict','wide','pair','power','all']:
   err=[]
   for ix in [np.arange(24),*[rng.permutation(24) for _ in range(3)]] if False else [np.arange(24),*[np.random.default_rng(seed).permutation(24) for seed in range(3)]]:
    f,k,e=mapB(u[ix[:20]],v[ix[:20]],kind,mode);err.append(np.mean(abs(f(u[ix[20:]])-v[ix[20:]])))
   f,k,e=mapB(u,v,kind,mode);results.append((np.mean(err),e,kind,mode,k))
 print('B fits', sorted(results)[:15]);
 for kind,mode in [('rat','strict'),('exp','wide'),('rat','power')]:
  f,k,e=mapB(u,v,kind,mode);R=v-f(u);print(kind,mode,'median,max',np.median(abs(R),axis=0),np.max(abs(R),axis=0)); print('coeff',np.round(ls(bias(np.c_[u,basis(u,kind,k)]),v),4))
 tr=s['train']; x=inv(tr['x'],calfit(s)); T=tr['traj'];y=tr['y']; f,*_=mapB(u,v,'rat','wide'); z=f(x); V=np.linalg.svd(T,full_matrices=False)[2];a=T@V[0]*V[0,-1];Y=np.c_[y,a,T@V[1],T@V[2]];print('y and trajectory mode correlations',np.corrcoef(Y.T));print('eval',s['eval_times'],s['telemetry_vec']);
 # Candidate rational denominators with numerator in the same calibrated operator frame.
 def lsD(z,d):return poly(z,d)[0]
 for key,w in [('B',z),('cal',x)]:
  F=lsD(w,2); r=Y[:,:2]-F@ls(F,Y[:,:2]); print(key,'quad coeff',np.round(ls(F,Y[:,:2]),3));print('largest resid',np.round(np.c_[w,Y[:,:2],r][np.argsort(-np.linalg.norm(r,axis=1))[:8]],3));
  for j in range(2):
   out=[];tar=Y[:,j];i=np.arange(56); q=np.arange(56,72)
   for deg in [1,2]:
    b=lsD(w,deg)
    def fit(res,b):return ls(b[i]/res[i,None],tar[i]);
    for n,h in [(str(l),w[:,l]) for l in range(4)]+[(f'{l}{sg:+}{m}',w[:,l]+sg*w[:,m]) for l,m in combinations(range(4),2) for sg in (-1,1)]:
     for shape in [h*h,abs(h)]:
      def loss(k):
       d=1+k*shape;c=fit(d,b);return np.mean((b[i]@c/d[i]-tar[i])**2)
      opt=minimize_scalar(loss,bounds=(0,6),method='bounded');d=1+opt.x*shape;err=np.mean(abs(b[q]@fit(d,b)/d[q]-tar[q]));out.append((err,opt.fun**.5,deg,n,opt.x))
   print('denominators',j,sorted(out)[:6],sorted(out,key=lambda r:r[1])[:4])
