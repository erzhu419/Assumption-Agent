import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; N=52
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
af=lambda x: np.c_[np.ones(len(x)),x]
a=s['panel_a']; d=af(a['u']); h=fit(d,a['v']); k=np.argsort(np.linalg.norm(d@h-a['v'],axis=1))[:41]; h=fit(d[k],a['v'][k]); z=(x-h[0])/np.diag(h[1:]); b=s['panel_b']; H=fit(af(b['u']),b['v']); w=af(z)@H
c=s['panel_c']; u=c['u']; Cs={}
for key in ('v1','v2'):
 v=c[key]; print('C panel',key)
 for name,fn in [('aff',af),('deg2',PolynomialFeatures(2).fit(u).transform),('aff+tanh',lambda q:np.c_[af(q),np.tanh(q)]),('aff+sin',lambda q:np.c_[af(q),np.sin(q)]),('aff+cos',lambda q:np.c_[af(q),np.cos(q)]),('aff+cube',lambda q:np.c_[af(q),q**3]),('aff+interact',lambda q:np.c_[af(q),*[q[:,i]*q[:,j] for i,j in combinations(range(4),2)]])]:
  f=fn(u); p=f@fit(f,v); lev=np.sum(f*np.linalg.pinv(f).T,axis=1); print(name,np.linalg.norm(p-v),np.mean(abs((p-v)/(1-lev[:,None]))));
for name,q in [('raw',x),('cal',z),('B',w),('Crot',z@fit(af(u),c['v1'])[1:]),('Crot2',z@fit(af(u),c['v2'])[1:])]:
 poly=PolynomialFeatures(3); f=poly.fit_transform(q); names=poly.get_feature_names_out(); p=fit(f[:N],y[:N]); o=abs(f[N:]@p-y[N:]); print(name,'full',o.mean(), 'coeffs',[(names[i],round(p[i],3)) for i in np.argsort(abs(p))[:-17:-1]])
 O=(poly.powers_.sum(1)!=2); p=fit(f[:N,O],y[:N]); print('odd',abs(f[N:,O]@p-y[N:]).mean(),[(names[O][i],round(p[i],3)) for i in np.argsort(abs(p))[:-13:-1]])
 R=y[:N]-af(q[:N])@fit(af(q[:N]),y[:N]); F=f[:,5:]; base=af(q); FR=F[:N]-base[:N]@fit(base[:N],F[:N]); FR=FR/np.maximum(np.linalg.norm(FR,axis=0),1e-12); keep=[]
 for _ in range(8):
  j=np.argmax(abs(FR.T@R));keep.append(j); v=np.c_[base,F[:,keep]]; p=fit(v[:N],y[:N]); R=y[:N]-v[:N]@p; print('omp',len(keep),names[5+j],round(abs(v[N:]@p-y[N:]).mean(),5),round(np.sqrt(np.mean(R**2)),5))
