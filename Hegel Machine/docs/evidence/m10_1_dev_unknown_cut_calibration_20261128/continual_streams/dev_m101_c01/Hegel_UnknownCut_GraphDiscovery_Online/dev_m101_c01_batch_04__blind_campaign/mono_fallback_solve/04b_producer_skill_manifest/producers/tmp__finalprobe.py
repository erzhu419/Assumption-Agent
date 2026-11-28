from operators import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations_with_replacement
np.set_printoptions(precision=5,suppress=True)
def poly(X,d):return PolynomialFeatures(d).fit_transform(X)
def cross(A,B):return (A[:,:,None]*B[:,None,:]).reshape(len(A),-1)
B=aff(x)@bb; Br=aff(raw)@bb
print('C2 coefficients',c2)
U,sv,V=np.linalg.svd(z,full_matrices=False); G=V[0]/V[0,-1]; end=z@G/(G@G)
# Additional covariate-tail slice checks extrapolation.
folds=folds+[(np.argsort(np.sum(raw**2,1))[:-14],np.argsort(np.sum(raw**2,1))[-14:])]
def cov(F):return F@F.T
def polyblocks(X):
 P=PolynomialFeatures(4); F=P.fit_transform(X); deg=P.powers_.sum(1); return [cov(F[:,deg==i]) for i in (3,4)]
K0=cov(poly(x,2)); blocks={'base':K0}
for k,X in [('A',x),('B',B),('Br',Br)]:
 blocks[k+'3'],blocks[k+'4']=polyblocks(X)
 blocks[k+'H']=cov(np.c_[np.sin(1.6*X),cross(np.sin(1.6*X),X)])
 blocks[k+'C1']=cov(np.c_[c1map(X),cross(X,c1map(X)),poly(c1map(X),2)[:,5:]])
 blocks[k+'C2']=cov(np.c_[cross(X,c2map(X)),poly(c2map(X),2)[:,5:]])
 for n,w in [('iso',np.full(4,.2)),('aniso',np.array([.4,.5,.25,.1]))]:
  D=1+(X*X)@w
  blocks[k+n]=cov(poly(X,3)/D[:,None])
 # Shift-free RBF and Matérn blocks capture remainder rather than replacing quadratic.
 D=np.sum((X[:,None]-X[None])**2,2)
 if k!='Br':
  for L in [1.,2.,4.]:blocks[k+'R'+str(L)]=np.exp(-D/(2*L*L))
def gp(k,tr,te,Y,floor):
 n=len(tr); A=k[:,tr[:,None],tr]; B=k[:,te[:,None],tr]; scale=np.diagonal(A,axis1=1,axis2=2).mean(1); A=A/scale[:,None,None]; B=B/scale[:,None,None]; I=np.eye(n); A=np.concatenate([A,I[None]]); sd=Y[tr].std(); Yt=Y[tr]/sd
 def obj(p,ret=False):
  w=np.exp(p); cf=cho_factor(np.einsum('i,ijk->jk',w,A)); al=cho_solve(cf,Yt)
  if ret:return sd*np.einsum('i,ijk,k->j',w[:-1],B,al)
  Q=cho_solve(cf,I)-np.outer(al,al); return .5*Yt@al+np.log(np.diag(cf[0])).sum(),.5*w*np.einsum('ijk,jk->i',A,Q)
 p=minimize(obj,np.r_[np.zeros(len(k))-2,-7],jac=True,method='L-BFGS-B',bounds=[(-15,6)]*len(k)+[(np.log(floor/sd**2),1)]).x
 return obj(p,True)
def fit_rat(X,F,tr,te,Y,mode):
 W=X*X
 if mode in (1,3):W=np.c_[W,cross(X,X)[:,[1,2,3,6,7,11]]]
 def pred(p,tgt):
  if mode in (1,3):
   L=np.array([[p[0],0,0,0],[p[4],p[1],0,0],[p[5],p[7],p[2],0],[p[6],p[8],p[9],p[3]]]); D=1+np.sum((X@L)**2,axis=1)
  else: D=1+W@p
  D=F/D[:,None]; M=D if mode<2 else np.c_[D,poly(X,2)]
  R=M[tr]; c=fit(R,Y[tr]) if mode<2 else np.linalg.solve(R.T@R+.0003*np.eye(R.shape[1]),R.T@Y[tr]); return M[tgt]@c
 init=np.r_[np.full(4,.4),np.zeros(6)] if mode in (1,3) else np.full(4,.2)
 p=least_squares(lambda p:pred(p,tr)-Y[tr],init,bounds=(0,5) if mode in (0,2) else (-3,3),max_nfev=35).x
 return pred(p,te)
sets=[['base','A3','B3','Br3','A4','B4','Br4','AC1','BC1','BrC1','AC2','BC2','BrC2','Aiso','Biso','Briso','Aaniso','Baniso','Braniso','AR1.0','AR2.0','AR4.0'],['base','A3','B3','BC2','BrC2','Biso','Briso','Baniso','Braniso'],['base','A3','B3','Br3','BC2','BrC2','AR1.0','AR2.0','AR4.0'],['base','A3','B3','Br3','A4','B4','Br4','BC2','BrC2'],['base','A3','B3','Br3','BC1','BrC1','BH','BrH','AR1.0','AR2.0','AR4.0'],['base','AR1.0','AR2.0','AR4.0'],['base','BC2','BrC2']]
for j,Y in enumerate((y,end)):
 preds=[];names=[]
 for i,keys in enumerate(sets):
  k=np.stack([blocks[key] for key in keys]);preds.append([gp(k,tr,te,Y,.00007 if j==0 else .000018) for tr,te in folds]);names.append('gp'+str(i))
 for a,X in [('A',x),('B',B),('Br',Br)]:
  for F,tag in [(poly(X,3),'poly3'),(np.c_[poly(X,2),cross(X,c2map(X))],'c2')]:
   for mode in range(4):
    if j==1 and mode>1:continue
    preds.append([fit_rat(X,F,tr,te,Y,mode) for tr,te in folds]);names.append((a,tag,mode))
 def score(pr):
  e=np.array([np.mean(abs(pp-Y[te])) for pp,(_,te) in zip(pr,folds)]);return np.round(np.r_[e[0],e[1:6].mean(),e[1:6].max(),e[-1]],5)
 order=sorted(range(len(names)),key=lambda i:np.mean(score(preds[i])[:3])); print('TARGET',j)
 for i in order[:15]:print(names[i],score(preds[i]))
 print('ensemble',score([np.mean(p,axis=0) for p in zip(*(preds[i] for i in order[:4]))]))
