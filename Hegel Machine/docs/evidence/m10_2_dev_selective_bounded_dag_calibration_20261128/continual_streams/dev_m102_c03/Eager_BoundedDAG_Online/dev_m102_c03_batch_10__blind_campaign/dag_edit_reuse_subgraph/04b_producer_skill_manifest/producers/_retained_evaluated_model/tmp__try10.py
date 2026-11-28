import pickle,numpy as np,itertools
from exact10 import latent,maps,ls
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; X=t['x']; z,st=latent(s,X); (b,c,d),misc=maps(s,z); print('A',st);print('B C',misc); print('latent',z.min(0),z.max(0));Y=np.c_[t['y'],t['traj']];q=-np.expm1(-s['eval_times'][:,None]*np.array([.4197,1.8143])); A=np.c_[t['y'],ls(q,t['traj'].T).T];
def dictionary(h,mode):
 u,v=h[:,[0,0,0,1,1,2]],h[:,[1,2,3,2,3,3]]; p=h[:,[0,0,0,1]]*h[:,[1,1,2,2]]*h[:,[2,3,3,3]]; zz=np.c_[h,u+v,u-v,u*v]; H=[h,h*h,h**3,u*v,u*u*v,u*v*v,p];
 if mode!='poly':
  for k in (.5,1.,1.25,1.5,1.8,2.15,2.4,3.,4.,5.,6.):
   H.extend((np.sin(k*zz),np.cos(k*zz),np.tanh(k*zz),zz/(1+k*zz*zz)))
  H.extend([u*v/(1+k*(u*u+v*v)) for k in (.22,.5,1.25,2.15,4)])
 return np.column_stack(H)
D={}; pairs=list(itertools.combinations(range(4),2)); l,r=z[:,[i for i,j in pairs]],z[:,[j for i,j in pairs]]
for tag,h in {'z':z,'raw':X,'b':b,'v1':c,'v2':d,'z+v1':z+c,'z+v2':z+d,'b+v1':b+c,'b+v2':b+d,'z+.22c':z+.22*c,'z+1.25c':z+1.25*c,'z+2.15c':z+2.15*c,'z+.22d':z+.22*d,'z+1.25d':z+1.25*d,'z-.22c':z-.22*c,'z-.22d':z-.22*d}.items():
 D[tag]=h

def err(pred,test):
 e=pred-A[test];return np.r_[np.mean(abs(e),0),np.mean(abs(e[:,1:]@q.T))]

def models(D,train,test):
 n=len(train); ix=np.r_[train,test]; W=A[ix]; m=W[:n].mean(0); T=W[:n]-m
 for tag,h in D.items():
  h=h[ix]; diff=h[:,None]-h[None,:n]; dist=np.sum(diff**2,axis=2)/max(np.median(np.sum(diff[:n]**2,axis=2)),1e-9)
  for k in (.2,.55,1.5,4,10):
   K=np.exp(-dist*k)
   for reg in (1e-6,1e-4,.003,.03): yield (tag,'rbf',k,reg),K[n:]@solve(K[:n]+reg*np.eye(n),T,assume_a='pos')+m
  for mode in ['poly','full']:
   F=dictionary(h,mode); F-=F[:n].mean(0); F/=np.maximum(np.linalg.norm(F[:n],axis=0),1e-12); B=F[:n]; K=B@B.T
   for reg in (.0001,.002,.015,.1,.5): yield (tag,mode,'ridge',reg),F[n:]@B.T@solve(K+reg*np.eye(n),T,assume_a='pos')+m
   if mode!='full':continue
   for key,rot in [('omp',False),('ols',True)]:
    P=np.zeros((28,len(test),3));
    for j in range(3):
     cols=[]; resid=T[:,j]; norms=np.ones(B.shape[1]); Q=np.empty((n,0));
     for p in range(28):
      scores=abs(resid@B)/np.maximum(norms,.05);scores[cols]=-1; k=np.argmax(scores);cols.append(k); G=B[:,cols]; coef=ls(G,T[:,j]);resid=T[:,j]-G@coef;P[p,:,j]=F[n:,cols]@coef+m[j];
      if rot:
       v=B[:,k]-Q@(Q.T@B[:,k]);v/=max(np.linalg.norm(v),1e-12);Q=np.c_[Q,v];norms=np.sqrt(np.maximum(1-np.sum((Q.T@B)**2,0),0));
    for p in (5,8,11,14,18,22,27): yield (tag,key,p+1),P[p]
R={}; splits=[(np.arange(52),np.arange(52,72))];
for j in range(3):
 t=np.random.default_rng(321+j).permutation(72);splits.append((t[:52],t[52:]));
for a,b_ in splits:
 for k,p in models(D,a,b_):R.setdefault(k,[]).append(err(p,b_));
for j in (0,1,2,3):
 for k,v in sorted(R.items(),key=lambda p:np.average(np.asarray(p[1])[:,j]))[:8]:print('best',j,k,np.round(np.asarray(v)[:,j],4));
