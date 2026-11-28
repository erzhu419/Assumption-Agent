import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
X,Y,T=(S['train'][k] for k in ('x','y','traj'))
def d(x):return np.c_[np.ones(len(x)),x]
def fit(A,B):return np.linalg.lstsq(A,B,rcond=None)[0]
def poly(x,k):return np.column_stack([np.prod(x[:,i],axis=1) for j in range(k+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
u,v=(S['panel_a'][k] for k in ('u','v')); keep=np.ones(len(u),bool)
for i in range(5):
 co=fit(d(u[keep]),v[keep]); r=np.linalg.norm(d(u)@co-v,axis=1);keep=r<3*np.median(r)
shift=co[0]; gain=np.diag(co[1:]); z=(X-shift)/gain
p=S['panel_b']; cb=fit(d(p['u']),p['v']); b=d(z)@cb
p=S['panel_c']; u=p['u']; D=lambda x:np.c_[d(x),x*x,x**3,x*np.roll(x,1,axis=1)]
np.set_printoptions(precision=4,suppress=True)
print('C1',fit(D(u),p['v1']));print('C2',fit(D(u),p['v2']));print('B',cb)
def operator(k):
 v=p[k]; models={'poly':lambda x,h:D(x),'quad':lambda x,h:poly(x,2),'Qcos':lambda x,h:np.c_[d(x),x*x,np.cos(h[0]*x)-1],'sin':lambda x,h:np.c_[d(x),np.sin(h[0]*x)],'cos':lambda x,h:np.c_[d(x),np.cos(h[0]*x)],'diagQcos':lambda x,h:np.c_[d(x),x*x+h[1]*(np.cos(h[0]*x)-1)]}
 scores={}
 for key,fun in models.items():
  def res(h,loo=False):
   a=fun(u,h); q=np.linalg.svd(a,full_matrices=False)[0]; r=a@fit(a,v)-v;return (r/(1-(q*q).sum(1)[:,None]) if loo else r).ravel()
  hs=[[1.5,.2],[2.8,-.3],[4,.3]]; h=min((least_squares(res,h,max_nfev=100).x for h in hs),key=lambda h:np.mean(res(h,True)**2));scores[key]=(np.sqrt(np.mean(res(h,True)**2)),np.sqrt(np.mean(res(h)**2)),h)
 print(k,scores)
 key=min(scores,key=lambda k:scores[k][0]);fun=models[key];h=scores[key][2];coef=fit(fun(u,h),v);return lambda x:fun(x,h)@coef
f,g=operator('v1'),operator('v2')
t=S['eval_times']; H=lambda l:-np.expm1(-t[:,None]*np.exp(l)); result=least_squares(lambda l:(H(l)@fit(H(l),T[:52].T)-T[:52].T).ravel(),np.log([.4,2.]),bounds=(-5,3)); h=H(result.x); amp=fit(h,T.T).T; print('rates',np.exp(result.x),'mode projection MAE',abs(amp@h.T-T).mean(0),'amplitude std',amp.std(0))
lat={'x':X,'z':z,'b':b,'z+b':z+b,'z-b':z-b,'b(x)':d(X)@cb}
res=[]
for k,q in lat.items():
 a,c=f(q),g(q)
 for name,F in {'affine':d(q),'poly2':poly(q,2),'poly3':poly(q,3),'C1':d(a),'C2':d(c),'C12':np.c_[d(a),c],'C12s':np.c_[d(a),c,(a*c).sum(1)],'C12p':np.c_[d(a),c,a*c],'C12qs':np.c_[d(a),c,(a*c).sum(1),(a*a).sum(1),(c*c).sum(1)],'C1poly':poly(a,2),'C2poly':poly(c,2),'C12raw':np.c_[d(q),a,c,a*c],'C12all':poly(np.c_[a,c],2)}.items():
  y=F@fit(F[:52],Y[:52]);s=F@fit(F[:52],amp[:52])@h.T; res.append((np.mean(abs(Y[52:]-y[52:])),np.mean(abs(T[52:]-s[52:])),np.mean(abs(Y[:52]-y[:52])),np.mean(abs(T[:52]-s[:52])),k,name))
print('best y');print(*sorted(res)[:15],sep='\n');print('best trajectory');print(*sorted(res,key=lambda r:r[1])[:10],sep='\n')
