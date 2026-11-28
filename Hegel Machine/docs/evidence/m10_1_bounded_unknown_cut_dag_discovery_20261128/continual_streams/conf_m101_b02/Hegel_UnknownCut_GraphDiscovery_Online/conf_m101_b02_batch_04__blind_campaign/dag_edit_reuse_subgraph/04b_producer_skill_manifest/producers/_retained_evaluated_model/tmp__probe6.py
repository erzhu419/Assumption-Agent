exec(open('/tmp/explore.py').read().split('funcs=')[0])
from scipy.linalg import solve
from scipy.spatial.distance import cdist
print('times',t,'keys',list(s),list(f));print('A keys',[ (k,np.shape(x)) for k,x in s['panel_a'].items()])
c=design(quad(z))@fit(design(quad(u)),s['panel_c']['v1']); d=design(z,np.sqrt(1+.93*z*z))@fit(design(u,np.sqrt(1+.93*u*u)),v)
z,b,c,d=[x[:n] for x in (z,b,c,d)]; Y=np.c_[y,traj]; folds=list(KFold(5,shuffle=True,random_state=13).split(y))+[(np.arange(n-20),np.arange(n-20,n))]
# Low-order interactions within and between calibration blocks.
R=[]
F=[z,b,c,d]
for i,j in folds:
 V=[(q-q[i].mean(0))/q[i].std(0) for q in F]; G=[q@q.T/q.shape[1] for q in V]; D=[cdist(q,q,'sqeuclidean')/q.shape[1] for q in V]
 R.append((i,j,G,D))
models=[]
for groups in [(0,1),(0,1,2),(0,1,3),(0,1,2,3),(1,2,3),(1,3)]:
 for wc in (.07,.25,1.):
  for ell in (.7,1.2,2.):
   for inter in (0.,.3,1.):
    for reg in (.00003,.0003,.003):
     scores=[]
     for i,j,G,D in R:
      w=[1 if g<2 else wc for g in groups]; gs=sum(a*G[g] for a,g in zip(w,groups)); ds=sum(a*D[g] for a,g in zip(w,groups)); A=sum(a*np.exp(-D[g]/(2*ell**2)) for a,g in zip(w,groups)); K=1+gs+A+inter*np.exp(-ds/(2*ell**2)); W=solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i],assume_a='pos'); E=abs(K[np.ix_(j,i)]@W-Y[j]); scores.append([E[:,0].mean(),E[:,1:].mean()]);
     A=np.array(scores); models.append((A.max(0).sum(),A.mean(0),A.max(0),A[-1],groups,wc,ell,inter,reg))
for row in sorted(models,key=lambda a:a[0])[:12]:print('add',row)
# Low complexity, reusable sparse polynomial across panel blocks, with weights selected inside the outer folds.
from sklearn.linear_model import ElasticNet,MultiTaskElasticNet
zc=np.c_[z,b,c,d]; cols=[(i,j) for i in range(16) for j in range(i,16)]; xx=np.c_[zc,*[zc[:,i]*zc[:,j] for i,j in cols],z**3,b**3,c**3,d**3]
for a in (.0003,.001,.003,.006,.012):
 for mix in (.5,.9,1.):
  rr=[]
  for i,j in folds:
   V=(xx-xx[i].mean(0))/(xx[i].std(0)+1e-8); A=ElasticNet(alpha=a,l1_ratio=mix,max_iter=16000).fit(V[i],y[i]); B=MultiTaskElasticNet(alpha=a,l1_ratio=mix,max_iter=16000).fit(V[i],traj[i]); rr.append([np.mean(abs(A.predict(V[j])-y[j])),np.mean(abs(B.predict(V[j])-traj[j]))]);
  rr=np.array(rr); print('sparse',a,mix,rr.mean(0),rr.max(0),rr[-1])
