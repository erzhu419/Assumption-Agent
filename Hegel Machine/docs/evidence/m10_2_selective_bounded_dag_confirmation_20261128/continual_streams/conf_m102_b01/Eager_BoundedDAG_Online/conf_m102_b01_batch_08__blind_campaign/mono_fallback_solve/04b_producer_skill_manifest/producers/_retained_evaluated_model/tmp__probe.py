import pickle,itertools,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X=tr['x']; Y=np.c_[tr['y'],tr['traj']]; P=lambda x,d=1:PolynomialFeatures(d).fit_transform(x); ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
a=s['panel_a']; A=P(a['u']); v=a['v'];
for j in range(4):
 q=least_squares(lambda q:A@q-v[:,j],ls(A,v[:,j]),loss='soft_l1',f_scale=.003).x; e=abs(A@q-v[:,j]); k=e<max(.004,8*np.median(e)); q=ls(A[k],v[k,j]); print('a',j,q,np.sort(e)[[0,23,36,40,43,46,47]])
 if j==0:M=[]
 M.append(q)
M=np.array(M).T; Z=(X-M[0])@np.linalg.inv(M[1:]); print('z range',Z.min(0),Z.max(0)); print('y identity trajectory',np.mean(abs(Y[:,0]-.5*(Y[:,1]+Y[:,-1]))));
D={'x':X,'z':Z};
for key,col in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[key]['u']; v=s[key][col]; name=key[-1]+col[1:];
 for d in (1,2,3):
  p=P(u,d); q=ls(p,v); print(name,d,'fit',np.mean(abs(p@q-v)),'loo',np.mean(abs((p@q-v)/np.maximum(1e-10,1-(p*np.linalg.pinv(p).T).sum(1))[:,None]))); D[name+str(d)]=P(Z,d)@q
 for f,iv,tag in [(np.arctanh,np.tanh,'tanh'),(np.arcsinh,np.sinh,'sinh'),(lambda v:v+v**3,lambda v:2/np.sqrt(3)*np.sinh(np.arcsinh(v*3*np.sqrt(3)/2)/3),'inv3')]:
  if tag=='tanh' and np.max(abs(v))>=1:continue
  for d in (1,2):
   p=P(u,d); q=ls(p,f(v)); print(name,tag,d,np.mean(abs(iv(p@q)-v))); D[name+tag+str(d)]=iv(P(Z,d)@q)
for k in list(D):
 if k not in ('x','z') and k[-1]!='3':D['z+'+k]=np.c_[Z,D[k]]
res=[]
def score(F,k,tag,reg):
 d=np.std(F[:52],axis=0);d[d<1e-9]=1; F=(F-F[:52].mean(0))/d; F=np.c_[np.ones(len(F)),F]; g=F[:52]; pred=F[52:]@np.linalg.solve(g.T@g+reg*np.eye(F.shape[1]),g.T@Y[:52]); e=abs(pred-Y[52:]); res.append((e[:,0].mean(),e[:,1:].mean(),k,tag,reg))
for k,x in D.items():
 for d in (1,2,3):
  if d==3 and x.shape[1]>4:continue
  for t,w in [('raw',x),('asinh',np.arcsinh(3*x)),('atan',np.arctan(2*x)),('inv3',np.sinh(np.arcsinh(5*x)/3)),('blend',np.c_[x,np.arcsinh(3*x)])]:
   if d==3 and t=='blend':continue
   for r in (.001,.03,.25,1):score(P(w,d)[:,1:],k,str(d)+t,r)
 x=(x-x[:52].mean(0))/np.maximum(x[:52].std(0),1e-9); dis=cdist(x,x[:52],'sqeuclidean')/x.shape[1]
 for l in (.5,1,2,4,8):
  for r in (.0001,.002,.02):
   for tag,K in [('rbf',np.exp(-.5*dis/l**2)),('mat',(1+np.sqrt(3*dis)/l)*np.exp(-np.sqrt(3*dis)/l))]:
    y=Y[:52].mean(0); pred=y+K[52:]@np.linalg.solve(K[:52]+r*np.eye(52),Y[:52]-y); e=abs(pred-Y[52:]); res.append((e[:,0].mean(),e[:,1:].mean(),k,tag+str(l),r))
for idx in (0,1):
 print('top for',idx);seen=set()
 for row in sorted(res,key=lambda p:p[idx]):
  k,tag=row[2:4]
  if (k,tag) in seen:continue
  seen.add((k,tag)); print(row)
  if len(seen)>=12:break
