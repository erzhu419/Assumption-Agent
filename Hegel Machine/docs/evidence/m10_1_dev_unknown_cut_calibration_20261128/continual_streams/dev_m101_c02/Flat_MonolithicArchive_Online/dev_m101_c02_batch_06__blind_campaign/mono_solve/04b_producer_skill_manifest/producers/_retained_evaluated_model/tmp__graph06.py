from model06 import *
from sklearn.model_selection import KFold
tr=s['train']; x=tr['x']; w=calibration(s)(x); y=tr['y']; T=tr['traj']; t=s['eval_times']; folds=[(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=4).split(w)]
def fitpanel(u,v):
 funcs={'rational':(lambda x,p:x**3/(1+p[0]*abs(x)**p[1])+p[2]*x,[[.5,2,-.1],[1,4,-.3]],([0,.5,-5],[12,9,5])), 'sine':(lambda x,p:x+p[0]*np.sin(p[1]*x),[[-.4,2.8],[.25,4],[-.12,7],[.2,9]],([-4,.2],[4,16])), 'two sine':(lambda x,p:x+p[0]*np.sin(p[1]*x)+p[2]*np.sin(p[3]*x),[[-.35,3,.03,6],[-.4,2,-.12,5],[.3,3,-.2,6],[-.3,2,.1,8]],([-4,.2,-4,.3],[4,15,4,18]))}
 candidates={}
 for name,(f,starts,bounds) in funcs.items():
  def a(x,p,f=f):return aff(f(x,p))
  def res(p):
   d=a(u,p);return (d@ls(d,v)-v).ravel()
  p=min((least_squares(res,q,bounds=bounds,max_nfev=150) for q in starts),key=lambda r:r.cost).x
  d=a(u,p); c=ls(d,v); e=np.sqrt(np.mean(res(p)**2)); candidates[name]=(e,lambda x,a=a,p=p,c=c:a(x,p)@c); print(name,np.round(p,4),round(e,6))
 return candidates
B=s['panel_b']; C=s['panel_c']; print('B'); models=fitpanel(B['u'],B['v']); print('C1'); models1=fitpanel(C['u'],C['v1']); b=models['rational'][1]; c1=min(models1.values(),key=lambda v:v[0])[1]; fq=lambda a:np.c_[aff(a),a*a]; cq=ls(fq(C['u']),C['v2']); c2=lambda a:fq(a)@cq

def report(name,z):
 a=aff(z); pair=np.c_[z[:,0]*z[:,2],z[:,1]*z[:,3]] if z.shape[1]==4 else np.empty((len(z),0)); ds=[a,np.c_[a,pair.sum(1)] if pair.shape[1] else a,np.c_[a,pair] if pair.shape[1] else a,poly(z,2) if z.shape[1]==4 else a]; result=[]
 for d in ds:
  errs=[]
  for i,j in folds:
   yp=d[j]@ls(d[i],y[i]); tp=d[j]@ls(d[i],T[i]); errs.append([abs(yp-y[j]).mean(),abs(tp-T[j]).mean()]);
  result.append(np.round(np.r_[errs[0],np.max(errs,axis=0)],5).tolist())
 print(name,result)
for n,q in models.items():report('B '+n,q[1](w))
z=b(w); p=c1(w); q=c2(w); bp=c1(z); bq=c2(z)
for name,g in [('C1',p),('C2',q),('B->C1',bp),('B->C2',bq),('C1->B',b(p)),('C2->B',b(q)),('parallel B,C1',np.c_[z,p]),('parallel B,C2',np.c_[z,q]),('serial C1 skip',np.c_[z,bp]),('serial C2 skip',np.c_[z,bq]),('B & raw/cal',np.c_[z,w]),('B,C1,C2',np.c_[z,p,q]),('B,C1(B),C2(B)',np.c_[z,bp,bq])]:report(name,g)
np.set_printoptions(precision=5,suppress=True); print('B train coordinates',z.std(0),z.min(0),z.max(0)); D=np.c_[aff(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]]; a=T@np.linalg.svd(T,full_matrices=False)[2][0]; l=ls(D,np.c_[y,a]); print('linear coefficients',l); e=np.c_[y,a]-D@l; print('resid correlations',np.corrcoef(np.c_[e,D[:,1:],w],rowvar=False)[:2]); print('largest residuals',np.c_[z,y,a,e][np.argsort(abs(e[:,0]))[-8:]])
