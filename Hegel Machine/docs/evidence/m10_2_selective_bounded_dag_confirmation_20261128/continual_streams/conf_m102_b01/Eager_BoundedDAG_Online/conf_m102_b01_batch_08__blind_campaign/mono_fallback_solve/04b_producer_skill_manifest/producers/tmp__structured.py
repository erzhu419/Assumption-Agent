import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=np.r_[s['train']['x'],s['test']['x']]; Y=np.c_[s['train']['y'],s['train']['traj']]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; one=lambda u:np.ones(len(u)); A=s['panel_a']; u,v=A['u'],A['v']; np.set_printoptions(precision=5,suppress=True)
def cal(u,j,d):
 a,b=u[:,j],u[:,(j+1)%4]; return np.column_stack([one(u),a,b,a*b]+[a**k for k in range(2,d+1)])
def robust(f,v):
 q=least_squares(lambda q:f@q-v,ls(f,v),loss='cauchy',f_scale=.015).x
 for _ in range(4):
  e=abs(f@q-v); keep=e<max(.001,6*np.median(e)); q=ls(f[keep],v[keep]);
 return q,e,keep
for d in range(1,9):
 out=[robust(cal(u,j,d),v[:,j]) for j in range(4)]; print('A',d,[round(np.sqrt(np.mean(e[k]**2)),6) for q,e,k in out],[sum(k) for q,e,k in out]);
 if d in (3,5,7): print(np.array([q for q,e,k in out]))
 if d==5: Q=[q for q,e,k in out]
Z=np.array([least_squares(lambda p:np.array([cal(p[None],j,5)[0]@Q[j] for j in range(4)])-x,x/1.2).x for x in X]); print('Z min max',Z.min(0),Z.max(0))
D={'X':X,'Z':Z}; c=s['panel_c'];
for deg in range(1,5):
 def f(u):return np.column_stack([one(u)]+[u**k for k in range(1,deg+1)])
 q=ls(f(c['u']),c['v2']); e=f(c['u'])@q-c['v2']; print('C2',deg,np.sqrt(np.mean(e**2,axis=0))); print(q) if deg<=3 else None
 D['C2'+str(deg)]=f(Z)@q
f=lambda u:np.c_[one(u),u*np.roll(u,-1,axis=1)]; D['C1']=f(Z)@ls(f(c['u']),c['v1']);
b=s['panel_b'];f=lambda u:np.c_[one(u),u];D['B']=f(Z)@ls(f(b['u']),b['v']);
for key in ['C1','C22','C23','B']:
 D['Z+'+key]=np.c_[Z,D[key]]
D['C1+C22']=np.c_[D['C1'],D['C22']]; D['Z+C1+C22']=np.c_[Z,D['C1'],D['C22']];
res=[]
for key,w in D.items():
 def score(f,reg,tag):
  f=f.copy(); f=(f-f[:52].mean(0))/np.maximum(f[:52].std(0),1e-9); f=np.c_[one(f),f]; g=f[:52]; p=f[52:72]@np.linalg.solve(g.T@g+reg*np.eye(f.shape[1]),g.T@Y[:52]); e=abs(p-Y[52:]); res.append([e[:,0].mean(),e[:,1:].mean(),key,tag,reg]);
 print('corr',key,np.round(np.corrcoef(np.c_[w[:72],Y[:,[0,1,-1]]].T)[:-3,-3:],2).T)
 for l in (.5,1,2,3,5,8):
  for tag,t in [('raw',w),('asinh',np.arcsinh(l*w)),('atan',np.arctan(l*w)),('inv3',np.sinh(np.arcsinh(l*w)/3)),('tanh',np.tanh(l*w))]:
   if tag=='raw' and l!=1:continue
   for kind,f in [('lin',t),('quad',PolynomialFeatures(2,include_bias=False).fit_transform(t)),('aug',np.c_[w,t])]:
    for r in (.001,.03,.25,1):score(f,r,(kind,tag,l))
 ww=(w-w[:52].mean(0))/np.maximum(w[:52].std(0),1e-8); dis=cdist(ww,ww[:52],'sqeuclidean')/w.shape[1]
 for l in (.5,1,2,4,8):
  for r in (.0001,.002,.02):
   for tag,K in [('rbf',np.exp(-.5*dis/l**2)),('mat',(1+np.sqrt(3*dis)/l)*np.exp(-np.sqrt(3*dis)/l))]:
    y=Y[:52].mean(0);pred=y+K[52:72]@np.linalg.solve(K[:52]+r*np.eye(52),Y[:52]-y); e=abs(pred-Y[52:]);res.append([e[:,0].mean(),e[:,1:].mean(),key,tag+str(l),r]);
for j in (0,1):
 print('best',j); seen=set();
 for r in sorted(res,key=lambda r:r[j]):
  if r[2] not in seen: print(r);seen.add(r[2])
np.savez('/tmp/operators.npz',**D,Y=Y)
