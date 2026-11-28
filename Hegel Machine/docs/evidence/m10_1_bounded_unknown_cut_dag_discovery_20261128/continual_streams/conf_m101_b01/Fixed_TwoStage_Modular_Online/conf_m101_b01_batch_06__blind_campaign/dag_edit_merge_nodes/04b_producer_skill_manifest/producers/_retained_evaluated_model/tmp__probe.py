from panels import *
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
p=s['panel_a']; u=p['u']; v=p['v']; A=lambda u:np.column_stack(list(features(u).values())[:-1]); c,e=robust(A(u),v); keep=np.argsort(e)[:-3]; c=np.linalg.lstsq(A(u)[keep],v[keep],rcond=None)[0]; print('A coef\n',c.round(4))
def cal(x):
 return np.array([least_squares(lambda a:A(a[None,:])[0]@c-r,np.linalg.solve(c[1:5].T,r-c[0])).x for r in x])
def score(F,Y):
 result=[]
 for a,b in folds:
  model=Ridge(alpha=.025).fit(F[a],Y[a]);result.append(np.mean(abs(Y[b]-model.predict(F[b])),axis=0))
 return np.array(result)
def panel_b(u):
 p=s['panel_b']; x,v=p['u'],p['v']; groups={'linear':lambda x,t:x,'sin':lambda x,t:np.sin(t*x),'tanh':lambda x,t:np.tanh(t*x),'soft':lambda x,t:x/(1+t*x*x),'atan':lambda x,t:np.arctan(t*x)}
 best=None
 for k,g in list(groups.items())[1:]:
  for k2,h in groups.items():
   if k==k2:continue
   def mat(x,t):return np.c_[np.ones(len(x)),g(x,t[0]),h(x,t[1])]
   def loss(t):
    M=mat(x,t);H=np.linalg.pinv(M);return ((M@(H@v)-v)/np.maximum(.01,1-np.sum(M*H.T,axis=1))[:,None]).ravel()
   opts=[least_squares(loss,a,bounds=(.05,4),max_nfev=40) for a in [(i,j) for i in [.5,1.3,2.8] for j in [.5,1.7,2.8]]]; opt=min(opts,key=lambda o:o.cost); print('B',k,k2,opt.x.round(4),np.sqrt(2*opt.cost/v.size).round(6));
   if best is None or opt.cost<best[0]:best=(opt.cost,mat,opt.x)
 _,m,t=best;M=m(x,t);b=np.linalg.lstsq(M,v,rcond=None)[0];print('B coef\n',b.round(3));return m(u,t)@b
x=s['train']['x'];u=cal(x); print('inverse |res| worst',abs(A(u)@c-x).max(),'latent range',u.min(axis=0),u.max(axis=0)); b=panel_b(u);p=s['panel_c'];C1=lambda x:np.c_[features(x)['lin'],features(x)['cross']]; C2=lambda x:np.c_[features(x)['lin'],np.log1p(.365*x*x)];d=C1(u)@np.linalg.lstsq(C1(p['u']),p['v1'],rcond=None)[0]; f=C2(u)@np.linalg.lstsq(C2(p['u']),p['v2'],rcond=None)[0];
z=s['train']['traj']; t=s['eval_times']; _,_,vh=np.linalg.svd(z,full_matrices=False); q=-vh[0]; print('time profile',q.round(4)); Y=np.c_[s['train']['y'],z@q*q[-1]];folds=list(KFold(4,shuffle=True,random_state=0).split(x))+[(np.arange(52),np.arange(52,72))];
R={'x':x,'u':u,'b':b,'c1':d,'c2':f}
for name,r in R.items():
 print('corr',name,np.corrcoef(np.c_[r,Y].T)[:-2,-2:].round(2));
 for deg in (1,2,3,4):
  F=PolynomialFeatures(deg,include_bias=False).fit_transform(r);out=score(F,Y);print(name,deg,out[:-1].max(0).round(4),out[-1].round(4))
for names in [('u','b','c1','c2'),('b','c1','c2'),('u','c1','c2')]:
 for deg in (1,2,3):
  F=np.column_stack([PolynomialFeatures(deg,include_bias=False).fit_transform(R[n]) for n in names]);out=score(F,Y);print(names,deg,out[:-1].max(0).round(4),out[-1].round(4))
np.savez('/tmp/reg.npz',x=x,u=u,b=b,c1=d,c2=f,Y=Y,q=q)
