exec(open('/tmp/readout_probe.py').read().split('res=[]')[0])
from scipy.optimize import least_squares
from sklearn.linear_model import ARDRegression,Lasso,ElasticNet
from itertools import combinations

def features(z,extended=False):
 s,g=arc['_features'](z); f=np.column_stack(s+g+[np.prod(z[:,j],axis=1) for j in combinations(range(4),3)])[:,1:]; cols=[q for q in f.T]; names=[str(i) for i in range(f.shape[1])];
 def add(n,v):names.append(n);cols.append(v)
 if extended:
  for i in range(4):
   u=z[:,i]
   for c in (.5,1,1.5,2,2.5,3):
    for n,fn in [('sin',np.sin),('cos',np.cos),('tan',np.tanh),('rat',lambda u:u/(1+u*u)),('abs',np.abs)]:add((i,n,c),fn(c*u))
  for i,j in combinations(range(4),2):
   u,v=z[:,i],z[:,j]
   for c in (.25,1,1.5,2):add((i,j,'ratio',c),u*v/(1+c*(u*u+v*v)))
   for d in (-2,-1,-.5,.5,1,2):
    w=u+d*v
    for n,fn in [('sin',np.sin),('cos',np.cos),('tan',np.tanh),('ss',lambda u:u/(1+abs(u))),('sym',np.abs)]:add((i,j,d,n),fn(w))
 return np.column_stack(cols),names

def cal(mask=None):
 p=st['panel_a'];u=p['u'];v=p['v'];h=np.c_[np.ones(len(u)),u]; keep=np.ones(len(u),bool)
 for _ in range(4):
  f=np.linalg.lstsq(h[keep],v[keep],rcond=None)[0];r=np.linalg.norm(v-h@f,axis=1);keep=r<max(1e-3,4*np.median(r))
 return (x-f[0])/f[1:].diagonal()

def mode(a):
 t=st['eval_times']; T=Y[a,1:].T
 def q(r):return -np.expm1(-t[:,None]*np.exp(r))
 def r(r):
  h=q(r);return (h@np.linalg.lstsq(h,T,rcond=None)[0]-T).ravel()
 o=least_squares(r,np.log([.4,1.8]));h=q(o.x);return np.c_[Y[a,0],np.linalg.lstsq(h,T,rcond=None)[0].T],h
if __name__=='__main__':
 for zi,z in enumerate([x,cal()]):
  for ex in (False,True):
   f,names=features(z,ex);allscore=[]
   for threshold in (3000,10000,30000,100000):
    ss=[]
    for a,b,_ in folds+[(np.arange(len(x)),np.arange(len(x)),None)]:
     u=(f-f[a].mean(0))/np.maximum(f[a].std(0),1e-9);A,Q=mode(a);P=[];SUP=[]
     for j in range(3):
      m=ARDRegression(threshold_lambda=threshold);m.fit(u[a],A[:,j]);P.append(m.predict(u[b]));SUP.append([(names[k],round(m.coef_[k],3)) for k in np.argsort(-abs(m.coef_))[:12] if abs(m.coef_[k])>.008]);
     P=np.array(P).T;E=abs(np.c_[P[:,0],P[:,1:]@Q.T]-Y[b]);ss.append([E[:,0].mean(),E[:,1:].mean()]);
    print(zi,ex,threshold,np.round(ss[0],4),np.max(ss[:-1],axis=0).round(4),np.mean(ss[:-1],axis=0).round(4),np.round(ss[-1],4));
    if threshold==10000:print(SUP)
