import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
np.set_printoptions(precision=4,suppress=True,linewidth=145)
u=s['panel_a']['u'];v=s['panel_a']['v']
for degree in (1,2,3):
 p=PolynomialFeatures(degree); F=p.fit_transform(u); names=p.get_feature_names_out()
 for j in range(4):
  q=least_squares(lambda q:F@q-v[:,j],np.zeros(F.shape[1]),loss='soft_l1',f_scale=.002,max_nfev=300).x
  e=abs(F@q-v[:,j]); keep=e<max(.0006,np.sort(e)[39]*1.4); q=ls(F[keep],v[keep,j]); e=abs(F@q-v[:,j]);print('a',degree,j,np.quantile(e,[.25,.5,.75,.8,.85,.9,1]),[(nm,round(c,4)) for nm,c in zip(names,q) if abs(c)>.008]);
for key,col in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[key]['u'];v=s[key][col];print(key,col,'range',v.min(0),v.max(0));p=PolynomialFeatures(2);F=p.fit_transform(u); q=ls(F,v);print(list(zip(p.get_feature_names_out(),q))); print('rms per col',np.std(F@q-v,0));
 for t in [.25,.5,.75,1.,1.25,1.5,2.,3.]:
  for typ in ['atanh','tan','sinh','asinh','inv3','logit']:
   w=v*t
   if typ=='atanh' and np.max(abs(w))>=1:continue
   if typ=='tan' and np.max(abs(w))>=np.pi/2:continue
   if typ=='logit' and np.min(w)<=-1:continue
   z={'atanh':lambda:np.arctanh(w),'tan':lambda:np.tan(w),'sinh':lambda:np.sinh(w),'asinh':lambda:np.arcsinh(w),'inv3':lambda:w+w**3,'logit':lambda:np.log1p(w)}[typ]()
   for d in (5,15):
    g=F[:,:d]; e=g@ls(g,z)-z; h=(g*np.linalg.pinv(g).T).sum(1); rm=np.sqrt(np.mean((e/(1-h)[:,None])**2,0))/np.std(z,0); 
    if rm.mean()<.05:print(t,typ,d,rm)
