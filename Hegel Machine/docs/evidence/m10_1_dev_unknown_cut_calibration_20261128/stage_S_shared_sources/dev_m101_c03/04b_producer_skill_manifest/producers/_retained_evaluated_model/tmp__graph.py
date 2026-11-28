import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
src=open('/tmp/campaign.py').read();exec(src[:src.index("u=s['panel_c']")]);
x,y,q=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times'];a=calfit(s['panel_a']); z=(x-a[:,0])/a[:,1]; p=s['panel_c'];u=p['u']
def panel_fit(u,v,which,kind):
 def phi(z,p): return .45*z+np.tanh(p*z) if which==1 else z+p*z*z
 def fit(par):
  f=phi(u,par); fm=f.mean(0); vm=v.mean(0); fc=f-fm; vc=v-vm
  if kind=='aff': c=ls(fc,vc)
  else:
   U,D,V=np.linalg.svd(fc.T@vc); c=(U@V)*D.sum()/np.sum(fc**2)
  d=vm-fm@c
  return lambda z:phi(z,par)@c+d,c,d
 p0=2.2 if which==1 else .45
 par=p0 if kind=='fixed' else least_squares(lambda k:(fit(k[0])[0](u)-v).ravel(),[p0],bounds=(.1,4)).x[0]
 f,c,d=fit(par);return f,(par,c,d)
def amps(q):
 def f(p):return -np.expm1(-t[:,None]*p)
 def e(p):
  F=f(p);return (F@ls(F,q.T)-q.T).ravel()
 rates=least_squares(e,[1.8,.4],bounds=(.015,10)).x;F=f(rates); return F,ls(F,q.T).T,rates
F,Y,rates=amps(q);print('rates',rates);Y=np.c_[y,Y];
for j in [1,2]:
 v=p[f'v{j}']; jj=np.arange(len(u))
 for kind in ['aff','orth','fixed']:
  f,k=panel_fit(u,v,j,kind); print('panel',j,kind,'par',k[0], 'sv',np.linalg.svd(k[1],compute_uv=False),'rms',rms(f(u)-v),'fold', [round(rms(panel_fit(u[jj%4!=k],v[jj%4!=k],j,kind)[0](u[jj%4==k])-v[jj%4==k]),5) for k in range(4)])
a=panel_fit(u,p['v1'],1,'orth')[0](z);b=panel_fit(u,p['v2'],2,'orth')[0](z)
P=PolynomialFeatures(2);zz=np.c_[a,b];M=P.fit_transform(zz);names=P.get_feature_names_out(['a','b','c','d','e','f','g','h']);
models={};models['linear']=M[:,:9];models['sameprod']=np.c_[M[:,:19],b,a*b];models['prev']=aff(np.c_[a,a*a,pair(a),b,a*b]);models['bothquad']=M;models['splitquad']=aff(np.c_[a,b,a*a,pair(a),b*b,pair(b)]);
folds=[]
for k in range(6):
 va=np.arange(len(x))%6==k;tr=~va;Tf,Yf,_=amps(q[tr]);folds.append((va,tr,Tf,np.c_[y[tr],Yf]))
for name,m in models.items():
 for alpha in [0,.00003,.0003,.003,.03]:
  ee=[]
  for va,tr,Tf,Yf in folds:
   h=m[va]@ls(m[tr],Yf) if alpha==0 else Ridge(alpha=alpha).fit(m[tr],Yf).predict(m[va]);ee.append(np.mean(abs(np.c_[h[:,0]-y[va],h[:,1:]@Tf.T-q[va]]),0))
  ee=np.array(ee);print(name,alpha,np.round([ee[:,0].mean(),ee[:,1:].mean(),ee[:,0].max(),ee[:,1:].max()],5))
for name,m,ns in [('prev',models['prev'],['1','a','b','c','d','a2','b2','c2','d2','ab','ac','ad','bc','bd','cd','e','f','g','h','ae','bf','cg','dh']),('bothquad',M,names)]:
 co=Ridge(alpha=.0002,fit_intercept=False).fit(m,Y).coef_.T;print(name,'coefficients',np.c_[ns,np.around(co,4)],'rms',np.std(m@co-Y,0))
