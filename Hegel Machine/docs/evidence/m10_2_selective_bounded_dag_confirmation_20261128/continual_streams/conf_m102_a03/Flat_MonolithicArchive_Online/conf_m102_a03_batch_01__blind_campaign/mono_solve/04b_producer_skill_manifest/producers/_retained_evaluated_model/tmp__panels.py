import numpy as np,pickle
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=150)
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
for p,outs in [('panel_a',['v']),('panel_c',['v1','v2'])]:
 u=s[p]['u']; f=PolynomialFeatures(2);D=f.fit_transform(u); names=f.get_feature_names_out()
 for out in outs:
  v=s[p][out]; ix=np.arange(len(u));
  for _ in range(15):
   C=fit(D[ix],v[ix]);e=np.linalg.norm(D@C-v,axis=1);ix=np.argsort(e)[:int(.84*len(u))]
  print(p,out,'coefficients');print('\n'.join(f'{n:6s} {c}' for n,c in zip(names,C)));print('row residuals',e)
  if p=='panel_a':
   for j in range(4):
    k=(j+1)%4
    print('channel',j, 'largest cubic correlations');res=v[:,j]-D@C[:,j];F=PolynomialFeatures(3);Q=F.fit_transform(u);R=Q-D@fit(D[ix],Q[ix]);score=abs(R[ix].T@res[ix])/(1e-12+np.linalg.norm(R[ix],axis=0)); ii=np.argsort(score)[-10:][::-1];print(list(zip(F.get_feature_names_out()[ii],score[ii])))
   print('panel a samples [u, v-u, residual]');print(np.c_[u,v-u,v-D@C])
print('train raw quantiles',np.quantile(s['train']['x'],[0,.1,.5,.9,1],axis=0))
