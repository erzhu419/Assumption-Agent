from graph import *
from scipy.special import erf
B=s['panel_b']; u=B['u']; v=B['v']; n=72
np.set_printoptions(precision=5,suppress=True)
def invc(x,a):
 t=x.copy()
 for _ in range(18):t-=(t+a*t**3-x)/(1+3*a*t*t)
 return t
def invt(x,a,b):
 t=x.copy()
 for _ in range(18):t-=(t+a*np.tanh(b*t)-x)/(1+a*b/np.cosh(b*t)**2)
 return t
families={'invc':(lambda x,a:invc(x,np.exp(a[0])),[-.4]),'invt':(lambda x,a:invt(x,*np.exp(a)),[-.4,.7]),'invtminus':(lambda x,a:x-a[0]*np.tanh(np.exp(a[1])*x),[.5,.4]),'saturating':(lambda x,a:x*(1+np.exp(a[0])*x*x)**a[1],[-.4,1.]),'erf':(lambda x,a:x+a[0]*erf(np.exp(a[1])*x),[-.5,.3]),'atan':(lambda x,a:x+a[0]*np.arctan(np.exp(a[1])*x),[-.5,.4]),'asin':(lambda x,a:x+a[0]*np.arcsinh(np.exp(a[1])*x),[-.5,.4]),'sinh':(lambda x,a:x+a[0]*np.sinh(np.exp(a[1])*x),[.3,-.5]),'sine':(lambda x,a:x+a[0]*np.sin(np.exp(a[1])*x),[-.4,1.05]),'cubic':(lambda x,a:x+a[0]*x**3+a[1]*x**5,[.5,0.]),'exponential':(lambda x,a:x+a[0]*x*np.exp(-np.exp(a[1])*x*x),[-.5,.3]),'softsign':(lambda x,a:x+a[0]*x/(1+np.exp(a[1])*abs(x)),[-.5,.3])}
for name,(fn,init) in families.items():
 d=lambda x,a:I(fn(x,a)); resid=lambda a:(d(u,a)@ls(d(u,a),v)-v).ravel()
 f=least_squares(resid,init,bounds=(-5,5),max_nfev=140); print('B fit',name,np.mean(abs(f.fun)),f.x); k=ls(d(u,f.x),v);ops[name]=lambda x,fn=fn,a=f.x,k=k:I(fn(x,a))@k
uc=c['u'];vc=c['v2'];d=lambda x,a:I(x+a[0]*x*x);r=lambda a:(d(uc,a)@ls(d(uc,a),vc)-vc).ravel(); f=least_squares(r,[.45]);print('C2 compressed',f.x,np.mean(abs(f.fun))); rop2=lambda x:d(x,f.x)@ls(d(uc,f.x),vc)
tr0=tr-tr[:52].mean(0);U,S,V=np.linalg.svd(tr0[:52],full_matrices=False); T=tr0@V.T;print('time',s['eval_times'],'S',S,'modes',V[:3],'mean',tr[:52].mean(0));print('Y correlation',np.corrcoef(np.c_[y,T].T)[0]);
def result(d,tt,k=52,lam=0):
 a,b=d[:k],d[k:72];scale=np.maximum(a.std(0),1e-6);scale[0]=1;a,b=a/scale,b/scale
 co=ls(a,tt[:k]) if not lam else np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@tt[:k]);return np.abs(b@co-tt[k:]).mean(0),b@co

def cross(a,b):return np.column_stack([np.sum(a*np.roll(b,k,axis=1),axis=1) for k in range(4)])
rows=[]
for bn in ['sin','sat','invc','invt','invtminus','sine','cubic','saturating']:
 w=ops[bn](z)
 for qi,q in [('w',qop(w)),('w+',qop(w)+ls(I(pair(uc)),c['v1'])[0]),('z',qop(z))]:
  for ri,r in [('z',rop2(z)),('w',rop2(w)),('q',rop2(q))]:
   blocks={'z':z,'w':w,'q':q,'r':r}; base=np.c_[z,q,r]; f1=np.c_[base,*[cross(a,b) for a,b in itertools.combinations(blocks.values(),2)]]
   f2=np.c_[base,outer(z,q),cross(q,r),cross(z,r),cross(w,q)]
   f3=np.c_[base,outer(z,q)]; f4=np.c_[base,outer(q,r)];f5=np.c_[base,outer(z,q),outer(z,r),outer(q,r),w];f6=np.c_[base,outer(z,q),outer(z,r),outer(q,r),w,quad(z),quad(q),quad(r)]
   for j,df in enumerate([f1,f2,f3,f4,f5,f6]):
    for lam in [.001,.03,.3,2]:
     dy,p=result(I(df),y,lam=lam); dt,pt=result(I(df),T[:,:3],lam=lam); rows.append((dy,*dt,bn,qi,ri,j,lam))
print('Y',*sorted(rows)[:12],sep='\n');print('T',*sorted(rows,key=lambda row:row[1])[:12],sep='\n'); print('T low modes',*sorted(rows,key=lambda row:sum(row[1:4]))[:10],sep='\n')
