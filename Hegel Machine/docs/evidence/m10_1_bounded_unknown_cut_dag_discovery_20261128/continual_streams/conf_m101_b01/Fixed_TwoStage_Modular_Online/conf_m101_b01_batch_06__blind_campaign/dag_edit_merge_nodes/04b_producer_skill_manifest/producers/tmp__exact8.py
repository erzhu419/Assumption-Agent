from panels import *
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
R=dict(np.load('/tmp/reg.npz')); u=R['u']; U,V=[s['panel_b'][k] for k in ('u','v')]; L=lambda x:np.c_[np.ones(len(x)),x]; best=[]
fun={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'asinh':np.arcsinh,'exp':lambda x:np.sign(x)*(-np.expm1(-abs(x))),'log':lambda x:np.sign(x)*np.log1p(abs(x))}
for name,g in fun.items():
 for mode in ('alone','lin','cubic'):
  def f(x,t):
   a=np.sign(x)*abs(x)**t[1];return np.column_stack([np.ones(len(x)),g(t[0]*a)]+({'alone':[],'lin':[x],'cubic':[x**3]}[mode]))
  def loss(t):
   m=f(U,t); p=np.linalg.pinv(m); return ((m@(p@V)-V)/(1-np.sum(m*p.T,axis=1))[:,None]).ravel()
  oo=[least_squares(loss,[k,3],bounds=([.005,1.5],[7,6]),max_nfev=80) for k in [.15,.65,1.5,3.]]; o=min(oo,key=lambda o:o.cost); coef=np.linalg.lstsq(f(U,o.x),V,rcond=None)[0];best.append((np.sqrt(np.mean(o.fun**2)),name,mode,o.x,coef,f(u,o.x)@coef))
best.sort(key=lambda r:r[0]);
for score,name,mode,t,coef,b in best[:8]:print('B',score,name,mode,t,coef.round(4) if score<.001 else '')
b=best[0][-1]; cb=best[0][4]; W={'u':u,'b':b,'bc':(b-cb[0]),'c1':R['c1'],'c2':R['c2'],'c1c':R['c1']-np.mean(R['c1'],0),'b_old':R['b']}; np.savez('/tmp/exact8.npz',**W)
y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; q=1-np.exp(-.55*t); Y=np.c_[y,z@q/(q@q)]; folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=0).split(u));
def scores(F):
 out=[]
 for a,h in folds:
  m=Ridge(alpha=.0005).fit(F[a],Y[a]);out.append(np.mean(abs(m.predict(F[h])-Y[h]),0))
 return np.array(out)
res=[]
for k,w in W.items():
 for n in (1,2,3,4):
  F=PolynomialFeatures(n,include_bias=False).fit_transform(w); v=scores(F);res.append((k,str(n),*v[0],*v[1:].max(0)))
 for den in ('axis','norm','pair'):
  if den=='axis': d=w*w
  if den=='norm': d=(w*w).sum(1)[:,None]
  if den=='pair': d=(w[:,i]+w[:,j])**2
  for a in (.3,1,3):
   P=PolynomialFeatures(2,include_bias=False).fit_transform(w); F=np.c_[P,(P[:,:,None]/(1+a*d[:,None,:])).reshape(len(w),-1)];v=scores(F);res.append((k,den+str(a),*v[0],*v[1:].max(0)))
for k,l in combinations(('u','b','c1','c2'),2):
 w,v=W[k],W[l]; F=np.c_[PolynomialFeatures(2,include_bias=False).fit_transform(w),PolynomialFeatures(2,include_bias=False).fit_transform(v),(w[:,:,None]*v[:,None,:]).reshape(len(w),-1)]; e=scores(F);res.append((k+'+'+l,'2',*e[0],*e[1:].max(0)))
for j in range(2,6):print('scores',j,sorted(res,key=lambda v:v[j])[:6]);
for k,x in W.items():
 P=PolynomialFeatures(3); F=P.fit_transform(x); co=np.linalg.lstsq(F,Y,rcond=None)[0]; e=Y-F@co; print('coeff',k,np.mean(abs(e),0),[[(P.get_feature_names_out()[a],round(co[a,j],4)) for a in np.argsort(abs(co[:,j])*F.std(0))[-8:][::-1]] for j in range(2)])
