from batch_solver import *
import itertools,pickle
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z,b=parts(s); tr=s['train']; n=72; T=tr['traj'];
splits=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=921).split(z[:n]));
pan=s['panel_c'];C=D(pan['u']*np.roll(pan['u'],-1,axis=1));c1=D(z*np.roll(z,-1,axis=1))@fit(C,pan['v1']);
G=lambda x:D(np.c_[x,x*x/(1+.4*x*x)]);c2=G(z)@fit(G(pan['u']),pan['v2']);
TF=[]
for i,j in splits:
 ss=dict(s,train={k:tr[k][i] for k in ('x','y','traj')}); f=times(ss);TF.append((f,np.c_[tr['y'],fit(f,T.T).T]));
def errors(out,j,f,Y):return np.array([np.mean(abs(out[:,0]-Y[j,0])),np.mean(abs(out[:,1:]@f.T-T[j])),np.mean(abs(out[:,-1]-Y[j,-1]))])
R=[]
for zc,bc,cf in itertools.product([.3,1,3],[.3,1,3],[0,.2]):
 for scale in [.3,1,3]:
  for act in [np.arcsinh,np.tanh,lambda x:x]:
   for mix in [0,1,2]:
    X=np.c_[zc*z,bc*b] if mix==0 else np.c_[zc*z+bc*b,zc*z-bc*b] if mix==1 else zc*z+bc*b
    if cf:X=np.c_[X,cf*c1,cf*c2]
    X=act(X*scale);L=(X@X.T)/X.shape[1];d=cdist(X,X,'sqeuclidean')/X.shape[1];
    for k in (.15,.7,2.5):
     for m in (0,1,2):
      K=(1+L)**2+np.exp(-k*d) if m==0 else (1+L)**3+np.exp(-k*d) if m==1 else (1+L)**2+(1+k*d)*np.exp(-k*d)
      for reg in (1e-6,.001,.02):
       es=[]
       for (i,j),(f,Y) in zip(splits,TF):
        h=K[j][:,i]@np.linalg.solve(K[i][:,i]+reg*np.eye(len(i)),Y[i]);es.append(errors(h,j,f,Y))
       e=np.array(es);R.append((e[0],e[1:].mean(0),e[1:].max(0),(zc,bc,cf,scale,act.__name__,mix,k,m,reg)))
for j in range(2):
 print('KERNEL',j,*sorted(R,key=lambda a:.45*a[0][j]+.4*a[1][j]+.15*a[2][j])[:7],sep='\n')
# Test separable nonlinear heads with variable projection and shared sum/difference coordinates.
R=[]
for fst in ['linear','quadratic','medium']:
 base=D(np.c_[z,b] if fst=='linear' else np.c_[z,P(b,2)[:,1:]] if fst=='quadratic' else np.c_[z,b,b*np.roll(b,1,axis=1),b*np.roll(b,2,axis=1)])
 for func in ['tanh','asinh','sin','rat','hinge','hinge2','cos','asym']:
  def H(x):
   if func=='tanh':return np.tanh(x)
   if func=='asinh':return np.arcsinh(x)
   if func=='sin':return np.sin(x)
   if func=='rat':return x/(1+x*x)
   if func=='hinge':return np.sign(x)*np.maximum(abs(x)-.35,0)**1.5
   if func=='hinge2':return np.sign(x)*np.maximum(abs(x)-.35,0)**2
   if func=='cos':return np.cos(x)
   return np.logaddexp(x,0)
  for rot in range(4):
   for mode in range(2):
    es=[]
    for (i,j),(f,Y) in zip(splits,TF):
     br=np.roll(b,rot,axis=1); B=base; slowF=np.c_[D(z),b,b*np.roll(b,1,axis=1),P(b,3)[:,-4:]]
     def F(q):
      w=z*q[0]+br*q[1];d=z*q[0]-br*q[1];return np.c_[B,H(w+q[2]),H(d+q[3])] if mode else np.c_[B,H(w+q[2])]
     def res(q):
      A=F(q)[i];return ((A@fit(A,Y[i,:2])-Y[i,:2])/np.std(Y[i,:2],0)).ravel()
     o=min((least_squares(res,[a,b0,.1,-.1],bounds=([.05,-6,-3,-3],[7,6,3,3]),max_nfev=65) for a,b0 in [(1,1),(1,-1),(2,2),(.5,2),(2,.5)]),key=lambda o:o.cost)
     A=F(o.x);h=np.c_[A[j]@fit(A[i],Y[i,:2]),slowF[j]@fit(slowF[i],Y[i,2])];es.append(errors(h,j,f,Y));
    e=np.array(es);R.append((e[0],e[1:].mean(0),e[1:].max(0),(fst,func,rot,mode)))
for j in range(2):
 print('SEPARABLE',j,*sorted(R,key=lambda a:.45*a[0][j]+.4*a[1][j]+.15*a[2][j])[:10],sep='\n')
