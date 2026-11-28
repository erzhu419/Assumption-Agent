import pickle,itertools,numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; tr=np.arange(52); va=np.arange(52,72)
ca,cb,cm=np.load('/tmp/panels.npy',allow_pickle=True); np.set_printoptions(precision=4,suppress=True,linewidth=160)
def one(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def capply(z,k):
 _,name,_,p,C=cm[k]; w=np.exp(p[0])*z+p[1:]; f=w/(1+abs(w)) if name=='softsign' else (np.cos(w) if name=='cos' else np.sin(w)); return one(np.c_[z,f])@C
ops={'A':lambda z:z*ca[1]+ca[0],'a':lambda z:(z-ca[0])/ca[1],'B':lambda z:one(z)@cb,'b':lambda z:(z-cb[0])@np.linalg.inv(cb[1:]),'C':lambda z:capply(z,0),'D':lambda z:capply(z,1)}
z={'x':x}
for dep in range(3):
 for name,v in list(z.items()):
  if len(name)!=dep+1:continue
  for op,f in ops.items():
   if op.swapcase()==name[-1] or op==name[-1]:continue
   z[name+op]=f(v)
H=-np.expm1(-t[:,None]*np.array([1.8,.4])); lat=ls(H,T.T).T; Y=np.c_[y,lat]; products=list(itertools.combinations(range(4),2)); Ysd=Y[tr].std(0)
def feats(v,mode):
 a=[v]
 if mode in ('p2','p3','all'):a += [v*v,np.stack([v[:,i]*v[:,j] for i,j in products],axis=1)]
 if mode in ('p3',):a += [v**3,np.stack([v[:,i]**2*v[:,j] for i in range(4) for j in range(4) if i!=j],axis=1),np.stack([np.prod(v[:,idx],axis=1) for idx in itertools.combinations(range(4),3)],axis=1)]
 if mode in ('sep','all'):a += [np.sin(v),np.cos(v),np.sin(2*v),np.cos(2*v),v/(1+abs(v)),np.tanh(v)]
 if mode=='all':
  w=np.stack([v[:,i]+j*v[:,k] for i,k in products for j in (-1,1)],axis=1); a += [np.sin(w),np.cos(w),np.tanh(w)]
 return np.column_stack(a)
def evaluate(F,alpha):
 m=F[tr].mean(0); sd=np.maximum(F[tr].std(0),1e-10); F=(F-m)/sd
 reg=Ridge(alpha=alpha).fit(F[tr],Y[tr]); P=reg.predict(F[va]); E=P-Y[va]; return np.r_[abs(E).mean(0),abs(E[:,1:]@H.T).mean()],np.sqrt(np.mean((reg.predict(F[tr])-Y[tr])**2,axis=0))
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import svd
from scipy.optimize import least_squares
T0=T.copy(); H=lambda p:-np.expm1(-t[:,None]*p); q=least_squares(lambda p:(T[tr]-ls(H(p),T[tr].T).T@H(p).T).ravel(),[1.8,.4]); Y=np.c_[y,ls(H(q.x),T.T).T]; print('rates',q.x)
S=z['xa']; print('summary compare',s['train']['summary'],[y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std()]);
for k in ['xa','xaC','xaD','xaB']:
 a=z[k]; print(k,'range',a.min(0),a.max(0))
D=z['xaD']; C=z['xaC']; B=z['xaB'];
for inp,name in [(D,'D'),(C,'C'),(np.c_[D,C],'DC')]:
 for powers in [2,3]:
  pp=PolynomialFeatures(powers,include_bias=True); f=pp.fit_transform(inp); nm=pp.get_feature_names_out();
  if name!='DC':f=np.c_[f,C if name=='D' else D]; nm=np.r_[nm,[('C' if name=='D' else 'D')+str(j) for j in range(4)]]
  norm=np.linalg.norm(f[tr],axis=0); A=f/norm
  for j in range(3):
   active=list(range(len(nm))); winner=[]
   while len(active)>4:
    aa=A[tr][:,active]; u,d,v=svd(aa,full_matrices=False); sh=d*d/(d*d+1e-6); co=v.T@((sh/d)*(u.T@Y[tr,j])); ev=A[:,active]@co-Y[:,j]; loo=(abs(ev[tr]/np.maximum(1-u**2@sh,1e-6))).mean(); mae=abs(ev[va]).mean(); winner.append((loo,mae,len(active),np.std(ev[tr]),list(zip(nm[active],np.round(co/norm[active],4))))); active.pop(int(np.argmin(np.abs(co))))
   w=min(winner,key=lambda x:x[0]); print(name,powers,j,w[:4],w[4] if w[2]<27 else '')
