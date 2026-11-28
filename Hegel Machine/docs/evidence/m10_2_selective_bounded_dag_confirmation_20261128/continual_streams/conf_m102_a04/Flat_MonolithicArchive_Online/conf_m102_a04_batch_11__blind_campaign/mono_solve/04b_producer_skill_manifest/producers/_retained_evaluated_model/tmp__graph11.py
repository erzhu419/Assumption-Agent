from struct11 import *
def calibration(raw):
 a=s['panel_a']; keep=np.linalg.norm(a['v'],axis=1)<5; u,v=a['u'][keep],a['v'][keep]; j=[2,3,0,1]
 def features(x,q):return np.stack([np.ones_like(x),x,np.sin(q[0]*x),np.sin(q[1]*x[:,j])],axis=2)
 def get(q):
  f=features(u,q); w=np.array([ls(f[:,i],v[:,i]) for i in range(4)]);return w,(f*w).sum(2)-v
 q=least_squares(lambda q:get(q)[1].ravel(),[2.4,2.4],bounds=([1,1],[4,4]),gtol=1e-13).x; w,e=get(q);print('CAL',q,np.mean(e**2)**.5,w.round(4)); return np.stack([least_squares(lambda x:(features(x[None,:],q)*w).sum(2)[0]-r,(r-w[:,0])/w[:,1],gtol=1e-12).x for r in raw])
def maps():
 p=s['panel_c']; u=p['u']; b=s['panel_b']; wb=ls(one(b['u']),b['v']); B=lambda w:one(w)@wb
 ff=lambda w,q:one(w/(1+q[0]*abs(w))+q[1]*w); gg=lambda w,q:np.c_[one(w),w*w,w*w/(1+q[0]*w*w)]
 q,w,*_=fit(ff,u,p['v1'],[.83,-.16],([.1,-.5],[4,.5])); r,h,*_=fit(gg,u,p['v2'],[.29],(.1,.5)); return B,lambda x:ff(x,q)@w,lambda x:gg(x,r)@h
def work():
 x=s['train']['x']; z=calibration(x); B,G,H=maps(); t=s['eval_times']; T=s['train']['traj']; ff=lambda t,q:-np.expm1(-t[:,None]*q); q,amp,*_=fit(ff,t,T.T,[.4,1.8],(.01,8)); F=ff(t,q); Y=np.c_[s['train']['y'],amp.T]; d={'x':x,'z':z,'b':B(z),'g':G(z),'h':H(z),'gb':G(B(z)),'hb':H(B(z)),'bg':B(G(z)),'bh':B(H(z))}; return d,Y,F
if __name__=='__main__':
 from sklearn.linear_model import Ridge,OrthogonalMatchingPursuit
 d,Y,T=work(); ids=list(d); S=s['train']; out=[]
 def score(f):
  m=f[:52].mean(0); sd=f[:52].std(0); f=(f-m)/np.maximum(sd,1e-9); best=[]
  for a in [1e-6,.01,.1,1,5,15]:
   p=Ridge(alpha=a).fit(f[:52],Y[:52]).predict(f[52:]);best.append((np.mean(abs(p[:,0]-Y[52:,0])),np.mean(abs(p[:,1:]@T.T-S['traj'][52:])),a));
  return min(best),min(best,key=lambda p:p[1])
 for nc in (1,2,3):
  for ii in it.combinations(ids,nc):
   a=[d[i] for i in ii]; P=np.column_stack(a); groups=[P];
   cross=[a[i]*a[j] for i,j in it.combinations(range(nc),2)]; allcross=[(a[i][:,:,None]*a[j][:,None,:]).reshape(-1,16) for i,j in it.combinations(range(nc),2)]; poly=np.column_stack([P[:,i]*P[:,j] for i,j in it.combinations_with_replacement(range(P.shape[1]),2)]); groups.extend([np.column_stack([P]+[v.sum(1) for v in []]+[c.sum(1) for c in cross]),np.column_stack([P,P**2]+cross),np.c_[P,poly],np.column_stack([P]+allcross)]);
   for k,g in enumerate(groups):
    a,b=score(g);out.append((a,b,ii,k))
 for k in [0,1]:
  print('TOP',k); print(*sorted(out,key=lambda v:v[k][k])[:18],sep='\n')
