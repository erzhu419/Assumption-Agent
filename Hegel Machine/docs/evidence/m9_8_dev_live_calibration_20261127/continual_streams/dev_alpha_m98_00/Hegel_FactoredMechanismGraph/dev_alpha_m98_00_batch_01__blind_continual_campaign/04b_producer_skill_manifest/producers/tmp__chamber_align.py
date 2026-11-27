from rheology_base import *
import itertools
P,H,X=pq; Y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; n=len(H)
T=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1])); C=np.linalg.lstsq(T,Y.T,rcond=None)[0].T
sets={'tr':tr,'te':te,'q':q}; cat=lambda gs,k:np.concatenate([sets[g][k] for g in gs]); I=np.arange(n); labs=q['lab_id']; missing=np.isnan(q['coupon_features_raw'])
def align(refset,fillset,method='median',idx=I):
 L=cat(refset,'lab_id'); R=cat(refset,'ref_coupons'); Lf=cat(fillset,'lab_id'); S=cat(fillset,'stratum'); F=cat(fillset,'coupon_features_raw').copy(); Q=q['coupon_features_raw'].copy()
 for l in np.unique(L):
  panel=R[L==l]; c=np.median(panel.mean(0)); s=np.median(panel.std(0));
  if method=='flat':c,s=panel.mean(),panel.std()
  if method=='row':c,s=np.median(panel.mean(1)),np.median(panel.std(1))
  F[Lf==l]=(F[Lf==l]-c)/s; Q[labs==l]=(Q[labs==l]-c)/s
 _,g,m=np.unique(np.c_[Lf,S],axis=0,return_inverse=True,return_counts=True); W=1/m[g,None]; fill=np.nansum(F*W,0)/np.sum(np.isfinite(F)*W,0)
 return np.where(missing,fill,Q)
def error(E):return np.array([np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),np.max(np.sqrt(np.mean(E[:,:-1]**2,axis=1))),np.max(abs(E[:,-1]))])
def cv(F,labels):
 E=np.zeros_like(Y)
 for l in np.unique(labels):
  k=labels!=l; E[~k]=F[~k]@np.linalg.lstsq(F[k],Y[k],rcond=None)[0]-Y[~k]
 return error(E)
parts=[p for r in (1,2,3) for p in itertools.combinations(sets,r) if p!=('tr',)]
results=[]
for ref,fill,method in itertools.product(parts,parts,['median','flat','row']):
 U=align(ref,fill,method); D=b[2]+U@b[3:]; F=np.c_[np.ones(n),H,D]; e=cv(F,I); results.append((e[0]+e[1],ref,fill,method,e,cv(F,labs)))
for _,r,f,m,e,g in sorted(results,key=lambda x:x[0])[:12]:print(r,f,m,'LOO',e.round(5),'LOLO',g.round(5))
for r,f,m in [(('q',),('q',),'median'),(sorted(results,key=lambda x:x[0])[0][1:4]),(('tr','te'),('tr','te'),'median')]:
 U=align(r,f,m); D=b[2]+U@b[3:]; F=np.c_[np.ones(n),H,D]; K=np.linalg.lstsq(F,C,rcond=None)[0]; print('alignment',r,f,m,'amplitude params',K.round(5),'CV full X',cv(np.c_[F[:,:2],U],I).round(5)); E=Y-F@K@T.T; _,sv,V=np.linalg.svd(E,full_matrices=False); print('residual sv and first signature',sv[:5].round(5),np.linalg.lstsq(T,V[0],rcond=None)[0].round(5)); print('residual with complete rows',error(E[~missing.any(1)]).round(5),'incomplete',error(E[missing.any(1)]).round(5)); print('amplitude differences vs H and D',np.c_[H,D,C[:,0]-b[0]*H,C[:,1]-b[1]*H,C[:,2]].round(4))
