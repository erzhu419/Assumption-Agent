import pickle,sys,numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y,T=tr['y'],tr['traj']; n=len(y); u,v=(s['panel_b'][k] for k in ('u','v')); w=m._calibrate(s['panel_a'],tr['x']); both=np.c_[y,T]
folds=[(np.arange(56),np.arange(56,n)),*KFold(4,shuffle=True,random_state=4).split(w)]; r=np.linalg.norm(w,axis=1); order=np.argsort(r); folds.append((order[:56],order[56:]));
def feature(q,p):
 h=q**3/(1+p[0]*q*q)
 for a,f in zip(p[1::2],p[2::2]):h+=a*np.sin(f*q)
 return np.c_[m._aff(q),h]
def score(P,j):
 e=abs(P-both[j]);return [e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]
for nh in (2,3):
 bounds=([.05]+[-.2,2]*nh,[3]+[.2,40]*nh)
 starts=[[.72,.008,10,.004,g] for g in (5,8,13,16,20,25,29,34)]
 if nh==3:starts=[p+[.002,g] for p in starts for g in (14,22,31,37) if abs(p[-1]-g)>2]
 def panel(p):
  A=feature(u,p); return A,m._ls(A,v)
 def pr(p):
  A,c=panel(p);return (A@c-v).ravel()
 fits=sorted((least_squares(pr,p,bounds=bounds,max_nfev=130) for p in starts),key=lambda o:o.cost); p=fits[0].x
 print('panel',nh,p,np.std(pr(p)),flush=True)
 _,c=panel(p); z=feature(w,p)@c
 for weight in (0,.20,.65):
  errors=[]; errors_r2=[]
  for i,j in folds:
   dy=m._design(w,z,'scalar'); x=dy[i]; cy=m._ls(x,y[i]); init=np.r_[0,cy[1:]]
   def link(d,b):
    h=b[0]+d[:,1:]@b[1:]; return np.c_[d[:,0],h+np.arctan(h)]
   def profile(P,output=False):
    p,b=P[:1+2*nh],P[1+2*nh:]; A,c=panel(p); Z=feature(w,p)@c; D=m._design(w,Z,'scalar'); X=m._design(w,Z,'trajectory'); d=link(D,b); cy=m._ls(d[i],y[i]); ct=m._ls(X[i],T[i]); pred=X@ct
    if output:return d[j]@cy,pred[j],pred[i]
    return np.r_[(A@c-v).ravel(), .45*weight*(d[i]@cy-y[i]), (weight/np.sqrt(T.shape[1]))*(pred[i]-T[i]).ravel()]
   if weight:
    B=([*bounds[0],*([-np.inf]*len(init))],[*bounds[1],*([np.inf]*len(init))]); param=min((least_squares(profile,np.r_[f.x,init*k],bounds=B,max_nfev=140) for f in fits[:2] for k in (1,2)),key=lambda r:r.cost).x; yp,tp,fit=profile(param,True)
   else:
    X=m._design(w,z,'trajectory'); fit=X@m._ls(X[i],T[i]); tp,fit=fit[j],fit[i]; yp=m._scalar(x,y[i],dy[j],'fixed')
   modes=np.linalg.svd(fit,full_matrices=False)[2][:2]; errors.append(score(np.c_[yp,tp],j));errors_r2.append(score(np.c_[yp,tp@modes.T@modes],j))
  for label,E in [('free',errors),('rank2',errors_r2)]: print(nh,weight,label,'56/16',np.round(E[0],6),'worst',np.round(np.max(E,0),6),'outer',np.round(E[-1],6),flush=True)
