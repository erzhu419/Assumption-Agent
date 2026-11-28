exec(open('/tmp/explore08.py').read().split('for name in')[0])
import conf_m102_a03_batch_02_solver as c
from itertools import combinations_with_replacement as cr
from scipy.optimize import minimize
F,G=c._panel_operators(s['panel_c']); h=m._warp(z,p); w=aff(h)@b; wc=w-b[0]; f=F(z); g=G(z)
sp=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))]; rng=np.random.default_rng(42); r=rng.permutation(72); sp.append((r[:52],r[52:]))
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def run(groups,penalties=(.001,.01,.1,1.),ret=False):
 out=[]
 for lam in penalties:
  rr=[]
  for i,j in sp:
   K=sum(np.divide(a,np.sqrt(np.mean(a[i]**2,0)).clip(1e-9))@np.divide(a[i],np.sqrt(np.mean(a[i]**2,0)).clip(1e-9)).T*wt/a.shape[1] for a,wt in groups); yp=K@np.linalg.solve(K[i]+lam*np.eye(len(i)),y[i]); rr.append(np.mean(np.abs(yp[j]-y[j])))
  out.append((np.round(rr,5),lam))
 return min(out,key=lambda p:np.mean(p[0]))
base=[(aff(z),5),(h,1),(cross(z,h),1),(pol(h,2)[:,4:],1)]
print('baseline',run(base)); print('w cross standardized',np.linalg.lstsq(np.c_[aff(z),wc,cross(z,wc),pol(wc,2)[:,4:]],y,rcond=None)[0].round(3));
features={}
for key,a in [('z',z),('h',h),('w',w),('wc',wc),('f',f),('g',g)]:
 for nm,func in [('square',lambda a:a*a),('cube',lambda a:a**3),('sat',lambda a:np.tanh(2*a)),('sine',lambda a:np.sin(3*a)),('abs',np.abs),('tfrac',lambda a:a/(1+a*a)),('qfrac',lambda a:a*a/(1+a*a)),('logcosh',lambda a:np.log(np.cosh(2*a)))]:
  d=func(a); features[key+nm]=d;features['z*'+key+nm]=cross(z,d); features['h*'+key+nm]=cross(h,d)
 for d in (2,3,4):
  P=np.column_stack([np.prod(a[:,js],axis=1) for js in cr(range(4),d)]);features[f'{key}poly{d}']=P;features[f'z*{key}poly{d}']=cross(z,P)
for k,a in [('f',f),('g',g),('w',w)]:
 for q,d in [('z',z),('h',h),('w',w),('f',f),('g',g)]:features[k+'*'+q]=cross(a,d)
rows=[]
for key,D in features.items():
 for wt in (.15,1,5):
  a,l=run(base+[(D,wt)],(.0001,.001,.01,.1));rows.append((np.mean(a),a,l,wt,key))
print('grammar additions to baseline',*sorted(rows,key=lambda a:a[0])[:24],sep='\n');print('by primary',*sorted(rows,key=lambda a:a[1][0])[:12],sep='\n')
# Search short coordinate-wise functions of the measured B outputs while retaining unrestricted lower-order interactions.
base2=np.c_[aff(z),w,cross(z,w),pol(w,2)[:,4:]]
for key,D in [('w',w),('z',z),('h',h),('f',f),('g',g)]:
 for kind in ('sin','tanh','frac','hinge'):
  def phi(D,q):
   if kind=='sin':return np.sin(q[0]*D+q[1]);
   if kind=='tanh':return np.tanh(q[0]*D+q[1]);
   if kind=='frac':return (D+q[1])/(1+q[0]*(D+q[1])**2)
   return np.maximum(D-q[1],0)**q[0]
  def fit(q,i,ret=False):
   P=phi(D,q); A=np.c_[base2,P]; A=A/np.sqrt(np.mean(A[i]**2,0)).clip(1e-9); t=A[i]; v=np.linalg.solve(t.T@t+.006*np.eye(t.shape[1]),t.T@y[i]); return A@v if ret else np.mean((t@v-y[i])**2)+.006*np.sum(v*v)/len(i)
  rr=[]
  for i,j in sp:
   opts=[minimize(fit,q,args=(i,),method='Nelder-Mead',bounds=[(.25,8),(-2,2)],options={'maxiter':100}) for q in [[1,0],[3,0],[5,.5]]]; best=min(opts,key=lambda a:a.fun); yy=fit(best.x,i,True);rr.append((round(np.abs(yy[j]-y[j]).mean(),5),best.x.round(3)))
  print('coord',key,kind,rr)
