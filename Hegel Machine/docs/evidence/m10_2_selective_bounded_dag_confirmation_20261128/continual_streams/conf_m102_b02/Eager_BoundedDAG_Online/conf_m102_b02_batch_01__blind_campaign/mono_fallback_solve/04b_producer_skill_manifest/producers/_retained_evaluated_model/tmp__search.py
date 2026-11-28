import numpy as np,pickle,itertools
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];Y=np.c_[s['train']['y'],s['train']['traj']];t=s['eval_times']; n=len(x)
def af(u):return np.c_[np.ones(len(u)),u]
def basis(u,kind):
 z=af(u);q=np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i,4)])
 if kind=='lin':return z
 if kind=='diag':return np.c_[z,u*u]
 if kind=='quad':return np.c_[z,q]
 if kind=='odd':return np.c_[z,u**3]
 if kind=='trig':return np.c_[z,np.sin(u),np.cos(u)]
 if kind=='tanh':return np.c_[z,np.tanh(u),np.tanh(u)**2]
 if kind=='cross':return np.c_[z,q[:,[1,2,3,5,6,8]]]
A=s['panel_a'];G=af(A['u']);i=np.ones(len(G),bool)
for _ in range(5):
 a=np.linalg.lstsq(G[i],A['v'][i],rcond=None)[0];r=np.linalg.norm(G@a-A['v'],axis=1);i=r<3*np.median(r)
z=np.linalg.solve(a[1:].T,(x-a[0]).T).T
ops={};kinds=['lin','diag','quad','odd','trig','tanh','cross']
for p,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u'];v=s[p][j]; name=p[-1]+j[-1];ops[name]={}
 for k in kinds:
  M=basis(u,k);P=np.linalg.pinv(M);w=P@v;er=M@w-v;loo=er/(1-(M*P.T).sum(1))[:,None];ops[name][k]=w
  print(name,k,np.round(np.sqrt(np.mean(er**2,0)),5),np.round(np.sqrt(np.mean(loo**2,0)),5))
 print('quad coefficients',np.round(ops[name]['quad'],4))
_,_,m=np.linalg.svd(Y[:,1:],full_matrices=False);print('TIME',t,'MODES',m[:2]);
for l in [0.5,0.75,1,1.25,1.5,2]:
 for f in [t,np.log1p(t),t*t,1-np.exp(-.22*t),1-np.exp(-2*t),np.ones(len(t))]:
  B=np.array([1-np.exp(-l*t),f]).T;er=Y[:,1:]-(Y[:,1:]@np.linalg.pinv(B).T)@B.T
  if np.mean(abs(er))<.018:print('time basis',l,f.round(3),np.mean(abs(er)))
Z={'x':z}
for o,dd in ops.items():
 for k,w in dd.items():
  if k in ('lin','diag','quad','cross','trig'):
   v=basis(z,k)@w;Z[o+k]=v;Z['x'+o+k]=np.c_[z,v]
for k in ['lin','diag','quad','cross','trig']:
 b=Z['bv'+k]; c=Z['c1'+k]; d=Z['c2'+k];Z['c12'+k]=np.c_[c,d];Z['all'+k]=np.c_[b,c,d];Z['xc12'+k]=np.c_[z,c,d];Z['bc12'+k]=np.c_[basis(b,k)@ops['c1'][k],basis(b,k)@ops['c2'][k]];Z['cb'+k]=basis(c,k)@ops['bv'][k];Z['csum'+k]=c+d;Z['cdiff'+k]=c-d
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=445).split(z));out=[]
for nm,v in Z.items():
 v=(v-v.mean(0))/v.std(0);D=cdist(v,v,'sqeuclidean')/v.shape[1];L=v@v.T/v.shape[1]; kernels={'linear':1+L}
 for p in [2,3,4]:kernels['poly'+str(p)]=(1+L)**p
 for a0 in [.08,.25,.7,1.5,4]:
  kernels['rbf'+str(a0)]=np.exp(-a0*D); kernels['rq'+str(a0)]=1/(1+a0*D); r=np.sqrt(a0*D);kernels['matern'+str(a0)]=(1+r+r*r/3)*np.exp(-r)
 for a0 in [.5,1,2]:
  h=np.arcsinh(a0*v);H=h@h.T/v.shape[1];kernels['asinh'+str(a0)]=(1+H+.2*L)**3+(h*h)@(h*h).T/v.shape[1];h=np.tanh(a0*v);kernels['tanh'+str(a0)]=(1+h@h.T/v.shape[1]+.2*L)**3
 for kn,K in kernels.items():
  for reg in [1e-6,1e-4,.002,.02]:
   e=[]
   for i,j in folds:
    pred=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i]);err=np.abs(pred-Y[j]);e.append([err[:,0].mean(),err[:,1:].mean()]);
   e=np.array(e);out.append((nm,kn,reg,*e[0],*e[1:].mean(0),*e[1:].max(0)))
for inds in [(3,4),(5,6),(5,),(6,)]:
 print('RANK',inds); seen=set()
 for v in sorted(out,key=lambda e:sum(e[i] for i in inds)):
  if v[0] in seen:continue
  seen.add(v[0]);print(v[:3],np.round(v[3:],4));
  if len(seen)>=12:break
