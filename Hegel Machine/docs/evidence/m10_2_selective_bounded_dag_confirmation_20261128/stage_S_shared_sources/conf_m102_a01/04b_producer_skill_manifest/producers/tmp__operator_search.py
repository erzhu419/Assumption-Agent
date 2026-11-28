import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; T=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; a=s['panel_a']; b=s['panel_b']; c=s['panel_c']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
aug=lambda u:np.c_[np.ones(len(u)),u]
lin=lambda u,v:np.linalg.lstsq(aug(u),v,rcond=None)[0]
m=lin(a['u'],a['v']); good=np.linalg.norm(a['v']-aug(a['u'])@m,axis=1)<2; A=lin(a['u'][good],a['v'][good]); B=lin(b['u'],b['v']); forward=lambda z,M:aug(z)@M; inverse=lambda z,M:(z-M[0])@np.linalg.inv(M[1:]);
for name in ('v1','v2'):
 u=c['u']; v=c[name]; print('\n',name)
 for d in range(1,6):
  F=np.column_stack([np.ones(len(u))]+[u**k for k in range(1,d+1)]); w=np.linalg.lstsq(F,v,rcond=None)[0]; h=np.sum(F*np.linalg.pinv(F).T,axis=1); print('separable',d,'rms',np.std(v-F@w),'loo',np.std((v-F@w)/(1-h[:,None])))
 P=PolynomialFeatures(2); F=P.fit_transform(u); w=np.linalg.lstsq(F,v,rcond=None)[0]; W=w[5:]@np.linalg.inv(w[1:5]); print('quadratic unmixed', list(zip(P.get_feature_names_out()[5:],W)))
 print('c matrices',w[:5]); print('diag ratios',np.diag(W[[0,4,7,9]]));
 for transform in (lambda u:u,lambda u:forward(u,A),lambda u:inverse(u,A),lambda u:forward(u,B),lambda u:inverse(u,B)):
  z=transform(u); cv=[]
  for func in (lambda z:z,lambda z:z*z,lambda z:np.tanh(z),lambda z:np.sin(z)):
   F=np.c_[aug(u),func(z)]; w=np.linalg.lstsq(F,v,rcond=None)[0]; cv.append(np.std(v-F@w));
  print('coordinate reduced residuals',cv)
E=lambda k:-np.expm1(-np.outer(t,k)); g=least_squares(lambda k:(T-T@np.linalg.pinv(E(k).T)@E(k).T).ravel(),[.22,1.52]); print('rates',g.x); Y=np.c_[y,T@np.linalg.pinv(E(g.x).T)];
for desc,z in [('raw',x),('a',forward(x,A)),('ai',inverse(x,A)),('b',forward(x,B)),('bi',inverse(x,B)),('aib',forward(inverse(x,A),B)),('ab',forward(forward(x,A),B)),('aibi',inverse(inverse(x,A),B))]:
 F=PolynomialFeatures(2).fit_transform(z); w=np.linalg.lstsq(F,Y,rcond=None)[0]; print(desc,'coefficients',w.T)
 for k in (6,8,10,12):
  errs=[]
  for j in range(3):
   cols=list(range(5));
   while len(cols)<k:
    choices=[]
    for i in set(range(F.shape[1]))-set(cols):
     ff=F[:52,cols+[i]]; ww=np.linalg.lstsq(ff,Y[:52,j],rcond=None)[0]; choices.append((np.linalg.norm(ff@ww-Y[:52,j]),i))
    cols.append(min(choices)[1]);
   ww=np.linalg.lstsq(F[:52,cols],Y[:52,j],rcond=None)[0]; errs.append(np.mean(abs(F[52:,cols]@ww-Y[52:,j])));
  print('sparse',k,errs)
