import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; ts=s['eval_times']; x=tr['x']; n=52
fit=lambda a,y:np.linalg.lstsq(a,y,rcond=None)[0]; one=lambda a:np.c_[np.ones(len(a)),a]
A,B,C=(s['panel_'+k] for k in 'abc'); u,v=A['u'],A['v']; keep=np.linalg.norm(v-u,axis=1)<4; cal=np.array([fit(one(u[keep,j,None]),v[keep,j]) for j in range(4)]).T; b=fit(one(B['u']),B['v'])
def fitop(kind,key):
 U=C['u']; V=C[key]
 def f(z,k):
  den=k[0]*abs(z) if kind==1 else k[0]*z*z
  return one(z+k[1]*((z if kind==1 else z*z)/(1+den)))
 def res(k):
  X=f(U,k);return (X@fit(X,V)-V).ravel()
 r=least_squares(res,[.8,-.8] if kind==1 else [.16,.54],max_nfev=500); w=fit(f(U,r.x),V);print('panel',key,r.x,np.std(r.fun),np.linalg.svd(w[1:])[1]); return lambda x: f(x,r.x)@w
c1=fitop(1,'v1');c2=fitop(2,'v2')
z=(x-cal[0])/cal[1]; bz=one(z)@b; bx=one(x)@b; charts={'x':x,'z':z,'bz':bz,'bx':bx,'c1z':c1(z),'c1b':c1(bz),'c2z':c2(z),'c2b':c2(bz),'bi':(z-b[0])@np.linalg.inv(b[1:])}
u,l,v=np.linalg.svd(T,full_matrices=False); print('y stats',y.min(),y.max(),y.mean(),y.std(),np.quantile(y,[.05,.25,.5,.75,.95]));print('x stats',x.min(0),x.max(0));print('trajectory baseline',np.mean(abs(T[n:]-T[:n].mean(0))), 'centered sigmas',np.linalg.svd(T-T.mean(0),compute_uv=False)); print('times and average',ts,T.mean(0)); a=T@v[0]; Y=np.c_[y,a];
splits=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=44).split(x)); J,K=np.triu_indices(4)
def scores(G):
 vals=[]
 for i,j in splits:
  d=G[np.ix_(j,i)]@cho_solve(cho_factor(G[np.ix_(i,i)]),Y[i])-Y[j]; vals.append(np.mean(abs(d),axis=0))
 return np.array(vals)
rec=[]
for name,g in charts.items():
 for trans in ['plain','atan','tanh','soft','sqrt','sin','rat']:
  for h in ([0] if trans=='plain' else [.2,.5,.85,1.25,2.]):
   f={'plain':lambda:g,'atan':lambda:np.arctan(h*g)/h,'tanh':lambda:np.tanh(h*g)/h,'soft':lambda:g/(1+h*abs(g)),'sqrt':lambda:g/np.sqrt(1+h*g*g),'sin':lambda:np.sin(h*g)/h,'rat':lambda:g/(1+h*g*g)}[trans](); F=f@f.T; P=cdist(f,f,'sqeuclidean'); D=cdist(g,g,'sqeuclidean'); L=g@g.T
   for q in [.1,1.]:
    for w in [0.,.04,.16,.45,1.2]:
     for r in [1.e-5,.001,.02]:
      G=1+F+q*F**2+r*np.eye(len(x)); G+=0 if not w else np.exp(-w*P); E=scores(G); rec.append((E[1:].mean(0),E[0],name,trans,h,q,w,r))
for j in range(2):
 print('TARGET',j)
 for order in ['oof','gate']:
  for V in sorted(rec,key=lambda v:v[0 if order=='oof' else 1][j])[:12]:print(order,V)
