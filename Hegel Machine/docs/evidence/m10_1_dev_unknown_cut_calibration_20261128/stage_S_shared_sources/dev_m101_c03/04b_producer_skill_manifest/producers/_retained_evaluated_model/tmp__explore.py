import numpy as np,pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x,y,q=[s['train'][k] for k in ['x','y','traj']]; t=s['eval_times']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def aff(u,v,rob=False):
 X=np.c_[np.ones(len(u)),u]; w=np.ones(len(u))
 for _ in range(15 if rob else 1):
  c=np.linalg.lstsq(X*w[:,None],v*w[:,None],rcond=None)[0];r=np.linalg.norm(v-X@c,axis=1);w=1/np.maximum(r,np.median(r)*1.5+1e-8)**.75
 if rob:
  ok=r<np.median(r)*6;c=np.linalg.lstsq(X[ok],v[ok],rcond=None)[0]
 return c
A=aff(**s['panel_a'],rob=True);B=aff(**s['panel_b']);C=aff(s['panel_c']['u'],s['panel_c']['v1']);print('A,B,C',A,B,C,sep='\n'); u=s['panel_a']['u']; print('A residual',np.sort(np.linalg.norm(u@A[1:]+A[0]-s['panel_a']['v'],axis=1))[[0,12,24,40,44,45,47]]);print('sv',*[np.linalg.svd(a[1:],compute_uv=False) for a in [A,B,C]])
print('times',t);U,D,V=np.linalg.svd(q,full_matrices=False);print('q temporal',V[:3]);print('q mean',q.mean(0))
for label, f in [('twoexp',lambda p:np.c_[1-np.exp(-p[0]*t),1-np.exp(-p[1]*t)]),('exp+time',lambda p:np.c_[1-np.exp(-p[0]*t),t]),('exp+osc',lambda p:np.c_[1-np.exp(-p[0]*t),np.sin(p[1]*t)]),('exp+const',lambda p:np.c_[1-np.exp(-p[0]*t),np.ones(len(t))])]:
 def err(p):
  F=f(p);return (np.linalg.lstsq(F,q.T,rcond=None)[0].T@F.T-q).ravel()
 z=least_squares(err,[1.,.3][:2 if label in ['twoexp','exp+osc'] else 1],bounds=(.0001,25));print(label,z.x,np.std(z.fun))
def get(a,u):return a[0]+u@a[1:]
def inv(a,u):return (u-a[0])@np.linalg.inv(a[1:])
a=get(A,x); ai=inv(A,x); b=get(B,x); c=get(C,x)
Z=np.c_[y,q@V[:2].T];Z=Z/Z.std(0)
for name,v in [('raw',x),('A',a),('Ai',ai),('B',b),('Bi',inv(B,x)),('C',c),('AB',get(B,a)),('AiB',get(B,ai)),('BA',get(A,b)),('AiBi',inv(B,ai)),('AC',get(C,a)),('AiC',get(C,ai)),('CAB',get(C,get(B,a))),('CAiB',get(C,get(B,ai))),('CAiBi',get(C,inv(B,ai)))]:
 P=PolynomialFeatures(4);F=P.fit_transform(v);nms=P.get_feature_names_out(['a','b','c','d']);m=F[:56].mean(0);sc=F[:56].std(0);sc[sc==0]=1;F=(F-m)/sc
 print(name,end=' ')
 for j in range(3):
  z=Z[:,j];d=[]
  for n in [4,8,12,18]:
   mdl=OrthogonalMatchingPursuit(n_nonzero_coefs=n).fit(F[:56],z[:56]);p=mdl.predict(F[56:]);d.append(round(np.mean(abs(p-z[56:])),4));
   if n==8: chosen=','.join(nms[mdl.coef_!=0])
  print(d,chosen,end=' | ')
 print()
print('rank score corr with x',np.corrcoef(np.c_[Z,x].T)[:3,3:]); print('score and y extremes',np.c_[y,q[:,0],q[:,-1],q@V[:2].T][np.argsort(y)[::3]])
