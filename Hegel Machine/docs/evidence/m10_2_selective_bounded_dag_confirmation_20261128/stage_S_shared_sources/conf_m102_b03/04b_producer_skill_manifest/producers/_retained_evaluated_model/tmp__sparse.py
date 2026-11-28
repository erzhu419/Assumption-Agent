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
from scipy.linalg import qr
print('telemetry',s['telemetry_vec'],'summary',s['train']['summary']); print('range',x.min(0),x.max(0),s['test']['x'].min(0),s['test']['x'].max(0))
print('panel coeff',ca,cb)
def build(keys,depth):
 D=[]; nm=[]
 def add(k,v):
  D.append(v); nm.extend([f'{k}[{j}]' for j in range(v.shape[1])])
 for k in keys:
  v=z[k]; add(k,v)
  if depth>=1:
   for label,op in [('sq',np.square),('cu',lambda w:w**3),('abs',abs),('ss',lambda w:w/(1+abs(w))),('rat',lambda w:w/(1+w*w)),('sin',np.sin),('tanh',np.tanh),('sin2',lambda w:np.sin(2*w)),('cos',np.cos),('cos2',lambda w:np.cos(2*w))]:add(label+k,op(v))
   for i,j in products:
    a,b=v[:,i:i+1],v[:,j:j+1]; add(f'{k}{i}*{j}',a*b)
    if depth>=2:
     for c in [-1,1]:
      w=a+c*b
      for label,op in [('s',np.sin),('ss',lambda w:w/(1+abs(w))),('t',np.tanh),('rat',lambda w:w/(1+w*w)),('c',np.cos)]:add(f'{label}({k}{i}+{c}*{j})',op(w))
   for i,j,l in itertools.combinations(range(4),3):add(f'{k}{i}*{j}*{l}',(v[:,i]*v[:,j]*v[:,l])[:,None])
 return np.column_stack(D),nm
def fit_omp(F,nm,base):
 for j in range(3):
  U=one(base); Q=qr(U[tr],mode='economic')[0]; sel=[]; record=[]
  for i in range(14):
   R=F[tr]-Q@(Q.T@F[tr]); norm=np.linalg.norm(R,axis=0); rr=Y[tr,j]-Q@(Q.T@Y[tr,j]); sc=abs(R.T@rr)/np.maximum(norm,1e-10); a=np.argmax(sc); sel.append(a); Q=np.c_[Q,R[:,a]/norm[a]]
   D=np.c_[U,F[:,sel]]; c=ls(D[tr],Y[tr,j]); err=D@c-Y[:,j]; record.append((abs(err[va]).mean(),np.std(err[tr]),i+1,[nm[k] for k in sel],c))
  print(j,sorted(record,key=lambda r:r[0])[:1]); print('early',[(a,b,c,d[-1]) for a,b,c,d,_ in record[:7]])
for key in [('x','xa','xaB','xaC','xaD','xaBC','xaBD'),('x','xa','xB','xC','xD','xBC','xBD'),('x','xa','xA','xaB','xaC','xaD','xaBC','xaBD','xABC','xABD','xaCB','xaDB','xAC','xAD')]:
 print(key); fit_omp(*build(key,2),x[:,0:0])
