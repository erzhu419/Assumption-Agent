import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def col(z):return np.c_[np.ones(len(z)),z]
def fit(u,v):
 a=col(u); m=np.ones(len(u),bool)
 for _ in range(5):
  c=ls(a[m],v[m]);e=np.linalg.norm(a@c-v,axis=1);m=e<max(.001,np.median(e)*3)
 return c
A=fit(**s['panel_a']); B=fit(**s['panel_b']); P=s['panel_c']; u=P['u']; V=P['v1']; W=P['v2'];
def fitsep(func,p,Y):
 def res(p):
  H=col(func(p,u));return (H@ls(H,Y)-Y).ravel()
 q=least_squares(res,p,gtol=1e-12,ftol=1e-11,max_nfev=150); c=ls(col(func(q.x,u)),Y);print('panel',q.x,np.std(q.fun));return lambda x:col(func(q.x,x))@c
C=fitsep(lambda p,z:np.tanh(p[0]*z)+p[1]*z,[2.2,.43],V)
CW=fitsep(lambda p,z:np.tanh(p[0]*z)+p[1]*z+p[2]*z**3,[2.2,.43,0],V)
D=fitsep(lambda p,z:z+p[0]*z**2+p[1]*z**4,[.53,-.064],W)
D2=fitsep(lambda p,z:z+p[0]*(1-np.cos(p[1]*z)),[1,1],W)
x=s['train']['x']; T=s['train']['traj']; y=s['train']['y']; _,_,v=np.linalg.svd(T[:52],full_matrices=False);Y=np.c_[y,T@v[:2].T];
sp=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.r_[0:20,40:72],np.arange(20,40))]
# Rank according to both independent output metrics.
R=[]
def score(n,F):
 ans=[]
 for a,b in sp:
  e=F[b]@ls(F[a],Y[a])-Y[b];ans.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@v[:2]))])
 ans=np.asarray(ans);R.append((ans.mean(),n,ans.round(5).tolist()))
a=(x-A[0])@np.linalg.inv(A[1:]); bf=lambda z:col(z)@B; bi=lambda z:(z-B[0])@np.linalg.inv(B[1:]);
ops={'B':bf,'Bi':bi,'C':C,'D':D,'tan':lambda z:np.tanh(1.8*z),'tans':lambda z:.3*z+.7*np.tanh(1.8*z),'Td':lambda z:np.tanh(z),'addCD':lambda z:C(z)+D(z),'subCD':lambda z:C(z)-D(z),'mulCD':lambda z:C(z)*D(z),'mixCD':lambda z:.7*C(z)+.3*D(z)}
poly=PolynomialFeatures(2); coords={}
def probe(n,z):
 coords[n]=z
 score(n+'/linear',col(z));score(n+'/quad',poly.fit_transform(z));score(n+'/partialquad',col(np.c_[z,z*z]));
 for w in (.7,1.4,2.2,3.2):
  score(n+'/tanh'+str(w),col(np.c_[np.tanh(w*z),z,z*z]))
  score(n+'/sin'+str(w),col(np.c_[np.sin(w*z),np.cos(w*z),z]))
for j,z in [('x',x),('a',a),('af',col(x)@A)]:
 probe(j,z)
 for k,op in ops.items():
  zz=op(z);probe(j+k,zz)
  score(j+k+'/Xz',col(np.c_[z,zz]));score(j+k+'/CD',col(np.c_[C(zz),D(zz)]));score(j+k+'/CDz',col(np.c_[zz,C(zz),D(zz)]))
  for l,op2 in ops.items():
   if k==l:continue
   zzz=op2(zz);probe(j+k+l,zzz)
   score(j+k+l+'/both',col(np.c_[zz,zzz]));score(j+k+l+'/CD',col(np.c_[C(zzz),D(zzz)]));score(j+k+l+'/CDz',col(np.c_[zzz,C(zzz),D(zzz)]))
for typ in [-1,0,1]:
 print('BEST',typ)
 for q in sorted(R,key=lambda r:r[0] if typ==-1 else np.mean(np.array(r[2])[:,typ]))[:12]:print(q)
