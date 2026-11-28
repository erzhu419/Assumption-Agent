import pickle,numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(x): return np.c_[np.ones(len(x)),x]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def quad(x,k):return np.column_stack([np.prod(x[:,j],axis=1) for j in combinations_with_replacement(range(4),k)])
def feats(u):
 return {'lin':D(u),'priorA':D(np.c_[u,u*np.roll(u,-1,axis=1),u**3]),'quad':D(np.c_[u,quad(u,2)]),'even':D(quad(u,2)),'quad_diag':D(np.c_[u,u**2]),'cubic_diag':D(np.c_[u,u**2,u**3]),'odd':D(np.c_[u,u**3,u**5]),'abs':D(np.c_[u,abs(u)]),'quad_cubdiag':D(np.c_[u,quad(u,2),u**3])}
def families():
 return {'tanh':(lambda u,p:np.tanh(p[0]*u),[1]),'sin':(lambda u,p:np.sin(p[0]*u),[2]),'erf':(lambda u,p:erf(p[0]*u),[1]),'threshold':(lambda u,p:np.sign(u)*np.maximum(abs(u)-p[0],0)**p[1],[.35,1.5]),'rat':(lambda u,p:u/(1+p[0]*u*u),[1]),'cos':(lambda u,p:np.cos(p[0]*u),[2]),'abs_pow':(lambda u,p:abs(u)**p[0],[2]),'tanh_offset':(lambda u,p:np.tanh(p[0]*(u-p[1])),[1,.2]),'sin_offset':(lambda u,p:np.sin(p[0]*(u-p[1])),[2,.2])}
if __name__=='__main__':
 for key in ['panel_a','panel_b','panel_c']:
  panel=s[key]; u=panel['u']
  for ky,v in panel.items():
   if ky=='u':continue
   ids=np.arange(len(u)); A=key=='panel_a'; keep=ids if not A else ids[np.argsort(np.linalg.norm(D(u)@fit(D(u),v)-v,axis=1))[:45]]
   print(key,ky,'range',u.min(),u.max()); results=[]
   def show(name,F,p=()):
    f=F[keep]; y=v[keep]; c=fit(f,y); e=f@c-y; lev=(f*np.linalg.pinv(f).T).sum(1); results.append((np.sqrt(np.mean((e/np.maximum(1-lev[:,None],1e-6))**2)),np.sqrt(np.mean(e**2)),name,np.round(p,4),c))
   for nm,F in feats(u).items(): show(nm,F)
   for nm,(h,p0) in families().items():
    for extra in ['none','linear']:
     def F(q):return D(h(u,q)) if extra=='none' else D(np.c_[u,h(u,q)])
     def err(q):
      f=F(q)[keep];return (f@fit(f,v[keep])-v[keep]).ravel()
     opts=[]
     for scale in [.5,1,1.7,2.6]:
      p=np.array(p0,dtype=float);p[0]*=scale; p[0]=min(p[0], .99 if nm=="threshold" else 9.99)
      opts.append(least_squares(err,p,bounds=([.001]+([-2] if len(p)==2 else []),[10]+([4] if len(p)==2 else [])) if nm!='threshold' else ([0,.2],[1.1,5]),max_nfev=100))
     op=min(opts,key=lambda o:o.cost); show(nm+' '+extra,F(op.x),op.x)
   for e,r,n,p,c in sorted(results,key=lambda z:z[0])[:8]:print('%.6f %.6f'%(e,r),n,p)
   print('winner coefficients',np.round(min(results,key=lambda z:z[0])[-1],4))
