from operators import *
from scipy.optimize import minimize_scalar
from scipy.linalg import qr
np.set_printoptions(precision=4,suppress=True,linewidth=130)
c=S['panel_c']; u=c['u']; families={'tanh':np.tanh,'sin':np.sin,'logcosh':lambda x:np.logaddexp(x,-x),'cos':np.cos,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'sigmoid':lambda x:x/np.sqrt(1+x*x)}
models={}
for j in ['v1','v2']:
 v=c[j]; fits=[]
 for key,h in families.items():
  def fit(a,loocv=True):
   f=np.c_[affine(u),h(a*u)]; b=ls(f,v); r=v-f@b
   if loocv:r/=1-(qr(f,mode='economic')[0]**2).sum(1)[:,None]
   return np.sqrt(np.mean(r*r)),b
  grid=np.linspace(.2,6,45); k=np.argmin([fit(a)[0] for a in grid]); a=minimize_scalar(lambda a:fit(a)[0],bounds=(grid[max(k-1,0)],grid[min(k+1,44)]),method='bounded').x
  fits.append((fit(a)[0],key,a,fit(a,False)[1]))
 for er,key,a,b in sorted(fits,key=lambda p:p[0])[:3]:
  print(j,key,a,er,'coef',b,'proportional',np.sum(b[1:5]*b[5:],1)/np.sum(b[1:5]**2,1))
 er,key,a,b=min(fits,key=lambda p:p[0]); models[j]=lambda w,h=families[key],a=a,b=b:np.c_[affine(w),h(a*w)]@b

def block(x):return np.c_[affine(x),powers(x,2)]
x=S['train']['x']; y=S['train']['y']; z=S['train']['traj']; t=S['eval_times']; Y=np.c_[y,ls(modes(t,z),z.T).T]; g=block(x)
print('conditional scalar');
for f in [affine(x), g, np.c_[g,powers(x,3)]]:
 for extra in [np.empty((len(x),0)),Y[:,1:]]:
  h=np.c_[f,extra]; b=ls(h,Y[:,0]); r=Y[:,0]-h@b; print(h.shape,'rms',np.std(r),'coeff end',b[-4:]);
b=ls(g,Y); print('quadratic residual corr',np.corrcoef((Y-g@b).T));
for i in range(3):
 H=np.zeros((4,4)); H[np.triu_indices(4)]=b[5:,i]; H=(H+H.T)/2; print('quad eig',i,np.linalg.eigh(H))
rng=np.random.default_rng(451); splits=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in [rng.permutation(72) for _ in range(4)]]; modes_fold=[modes(t,z[tr]) for tr,te in splits]
results=[]
for k,w in coords(x,S).items():
 c1,c2=(models[j](w) for j in ['v1','v2']);
 tests={'c1c2':np.c_[np.ones(len(x)),c1,c2],'w-c1':np.c_[affine(w),c1],'w-c2':np.c_[affine(w),c2],'w-c1c2':np.c_[affine(w),c1,c2],'c1-p2':block(c1),'c2-p2':block(c2),'c1-p2-c2':np.c_[block(c1),c2],'c2-p2-c1':np.c_[block(c2),c1],'w-p2-c1c2':np.c_[block(w),c1,c2],'w-c1c2-diag':np.c_[affine(w),c1,c2,w*c1,w*c2,c1*c2]}
 for name,h in tests.items():
  es=[]
  for m,(tr,te) in zip(modes_fold,splits):
   p=h[te]@ls(h[tr],np.c_[y[tr],ls(m,z[tr].T).T]); es.append([np.mean(abs(p[:,0]-y[te])),np.mean(abs(p[:,1:]@m.T-z[te]))]);
  results.append((k,name,np.array(es)))
for j in range(2):
 for k,name,e in sorted(results,key=lambda p:p[2][:,j].mean())[:8]: print(j,k,name,'gate',e[0],'mean',e.mean(0),'worst',e.max(0))
