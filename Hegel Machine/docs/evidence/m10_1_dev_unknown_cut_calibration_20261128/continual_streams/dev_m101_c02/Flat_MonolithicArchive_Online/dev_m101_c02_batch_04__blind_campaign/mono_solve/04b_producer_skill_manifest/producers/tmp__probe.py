import pickle, numpy as np
from itertools import combinations, combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x: np.c_[np.ones(len(x)),x]
def pol(x,d): return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in combinations_with_replacement(range(4),k)])
for pn in ['panel_a','panel_b','panel_c']:
 p=s[pn]; u=p['u'];
 for key,v in p.items():
  if key=='u': continue
  if pn=='panel_a':
   good=np.max(abs(v-u),axis=1)<3; u,v=u[good],v[good]; eye=np.eye(4); nxt=np.roll(u,-1,axis=1); X=np.c_[np.tile(eye,(len(u),1)),(u[:,:,None]*eye).reshape(-1,4),u.ravel()**3,((u-1)*nxt).ravel()]; b=fit(X,v.ravel()); print('archive A fit',b,np.sqrt(np.mean((X@b-v.ravel())**2))); print('A generic quadratic/cubic train RMS',*[np.sqrt(np.mean((pol(u,d)@fit(pol(u,d),v)-v)**2)) for d in (2,3)])
  for name,f in [('linear',aff),('sq',lambda x:np.c_[aff(x),x*x]),('quad',lambda x:pol(x,2)),('sin',lambda x:np.c_[aff(x),np.sin(x),np.cos(x)]),('sq+cubic',lambda x:np.c_[aff(x),x*x,x**3]),('cyc',lambda x:np.c_[aff(x),x*x,x*np.roll(x,-1,axis=1)]),('opp',lambda x:np.c_[aff(x),x*x,x[:,:2]*x[:,2:]])]:
   X=f(u); H=X@np.linalg.pinv(X); r=v-H@v; print(pn,key,name,'train/LOO RMS',*(round(np.sqrt(np.mean(z*z)),6) for z in (r,r/(1-H.diagonal()[:,None]))))
