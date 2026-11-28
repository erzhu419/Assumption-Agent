import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def aff(u):return np.c_[np.ones(len(u)),u]
def poly(u,k):return np.column_stack([np.prod(u[:,i],axis=1) for j in range(k+1) for i in itertools.combinations_with_replacement(range(u.shape[1]),j)])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
a=s['panel_a']; U=aff(a['u']); V=a['v']; ii=np.arange(len(U));
for _ in range(6):
 wa=fit(U[ii],V[ii]); r=np.linalg.norm(U@wa-V,axis=1); ii=np.argsort(r)[:40]
print('cal',wa,'resid',np.sort(r));
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[pn]; u=p['u']; v=p[vn]; print(pn,vn,'ranges',u.min(0),u.max(0),v.min(0),v.max(0)); print('first rows',np.c_[u,v][:4]);
 fs={'linear':aff(u),'quad':poly(u,2),'lin+sinu':np.c_[aff(u),np.sin(u)],'quad+sinu':np.c_[poly(u,2),np.sin(u)],'linear+tanh':np.c_[aff(u),np.tanh(u)],'linear+sq':np.c_[aff(u),u*u],'linear+cube':np.c_[aff(u),u**3]}
 for k,f in fs.items():
  lev=(f*np.linalg.pinv(f).T).sum(1); e=v-f@fit(f,v); print(k,np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-lev[:,None]))**2)))
 for scale in [0.5,0.65,0.8,1.,1.2,1.5,2.]:
  if np.max(abs(v))<scale:
   z=np.arctanh(v/scale); print('atanh',scale,*[round(np.std(z-f@fit(f,z)),5) for f in [aff(u),poly(u,2)]])
