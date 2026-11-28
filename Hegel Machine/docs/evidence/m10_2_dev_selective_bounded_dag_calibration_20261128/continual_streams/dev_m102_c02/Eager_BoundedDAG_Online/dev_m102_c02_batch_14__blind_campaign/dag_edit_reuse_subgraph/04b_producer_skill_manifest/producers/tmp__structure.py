import numpy as np,pickle,itertools
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=170)
a=s['panel_a']; u=a['u']; v=a['v']; good=np.max(abs(v-u),axis=1)<2; print('A rows removed',np.where(~good)[0]);
def probe(x,y,name,deg=2):
 P=PolynomialFeatures(deg); d=P.fit_transform(x); n=P.get_feature_names_out(); co=np.linalg.lstsq(d,y,rcond=None)[0]; r=y-d@co; h=(np.linalg.qr(d,mode='reduced')[0]**2).sum(1); print(name,'mae',abs(r).mean(0),'loo',abs(r/(1-h[:,None])).mean(0));
 for j in range(y.shape[1]):
  jj=np.argsort(abs(co[:,j]))[::-1][:12]; print(j,list(zip(n[jj],co[jj,j].round(4))))
 return d,co
for deg in [1,2,3]: probe(u[good],v[good],f'A{deg}',deg)
for panel,src,tgt in [('panel_b','u','v'),('panel_c','u','v1'),('panel_c','u','v2'),('panel_c','v1','v2'),('panel_c','v2','v1')]:
 p=s[panel]; probe(p[src],p[tgt],panel+src+tgt)
# See whether panel B is described by the same map measured on panel C.
b=s['panel_b']; c=s['panel_c']; P=PolynomialFeatures(2); d=P.fit_transform(np.r_[b['u'],c['u']]);
for key in ['v1','v2']:
 yy=np.r_[b['v'],c[key]]; r=yy-d@np.linalg.lstsq(d,yy,rcond=None)[0]; print('joint B,C',key,abs(r).mean(0));
