import numpy as np,pickle
from itertools import combinations_with_replacement as cr,combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; n=len(x)
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
ix=[j for k in range(1,4) for j in cr(range(4),k)]
def des(z,inds=ix):return np.column_stack([np.ones(len(z))]+[np.prod(z[:,j],axis=1) for j in inds])
u,v=map(s['panel_a'].get,['u','v']); H=des(u); A=H[:,:5]; keep=np.linalg.norm(v-A@fit(A,v),axis=1)<2; W=fit(H[keep],v[keep]); print('A inliers',sum(keep)); print([(j,w.round(4).tolist()) for j,w in zip([()]+ix,W) if max(abs(w))>.009]);
# Recover each calibration channel's sparse exact polynomial, then refit jointly.
B=W.copy()
for j in range(4):
 idx=np.flatnonzero(abs(W[:,j])>.015); B[:,j]=0; B[idx,j]=fit(H[keep][:,idx],v[keep,j]); print('channel',j,'resid',np.std((H@B-v)[keep,j]),'terms',[( ([()]+ix)[i],round(B[i,j],5)) for i in idx]);
linear=B[:5]; start=(x-linear[0])@np.linalg.inv(linear[1:]); z=np.array([least_squares(lambda q:des(q[None,:])[0]@B-row,z0).x for row,z0 in zip(x,start)]); print('inverse residual',abs(des(z)@B-x).max(),'extrema',z.min(0),z.max(0));
u=s['panel_c']['u']; R={}
for k in ('v1','v2'):
 v=s['panel_c'][k]; fn=(lambda z,a:np.tanh(a*z)) if k=='v1' else (lambda z,a:np.expm1(-a*z*z)); h=lambda a:np.c_[des(u)[:,:5],fn(u,a)]; er=lambda a:(v-h(a)@fit(h(a),v)).ravel(); a=least_squares(er,np.repeat(2.2 if k=='v1' else .29,4),bounds=(.01,8)).x; w=fit(h(a),v); print(k,'nonlinear params',a,'rms',np.std(er(a)),'mix',w[5:]@np.linalg.inv(w[1:5]));R[k]=(a,w,fn)
p=s['panel_b']; PB=fit(des(p['u'])[:,:5],p['v']); Z={'z':z,'rot':z@PB[1:]/.88}
for k,(a,w,fn) in R.items():
 Z[k]=z+fn(z,a)@w[5:]@np.linalg.inv(w[1:5]); Z[k+'map']=np.c_[des(z)[:,:5],fn(z,a)]@w
rng=np.random.default_rng(1); splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(rng.permutation(n),4)]; Q=-np.expm1(-t[:,None]*[.408,1.815]); targets=np.c_[Y[:,0],fit(Q,Y[:,1:].T).T];
def score(pr,j):
 E=pr-targets[j]; return np.array([np.mean(abs(E[:,0])),np.mean(abs(pr[:,1:]@Q.T-Y[j,1:]))]);
def ridge(F,reg):
 ans=[]
 for i,j in splits:
  m=F[i].mean(0); sd=np.maximum(F[i].std(0),1e-7); A=(F[i]-m)/sd; D=(F[j]-m)/sd; ym=targets[i].mean(0); ans.append(score(D@np.linalg.solve(A.T@A+reg*np.eye(F.shape[1]),A.T@(targets[i]-ym))+ym,j))
 return np.array(ans)
def lib(z):
 f=list(des(z).T[1:]); names=[str(j) for j in ix]
 def add(v,name): f.append(v); names.append(name)
 for i in range(4):
  u=z[:,i];
  for a in (.5,1,1.5,2,2.2,3):
   for fn in (np.sin,np.cos,np.tanh,np.arctan):add(fn(a*u),f'{fn.__name__}({a}*{i})')
  for a in (.3,.5,1,2):add(np.exp(-a*u*u),f'exp({a}*{i}^2)'); add(u/(1+a*u*u),f'rat({a}*{i})')
 for i,j in combinations(range(4),2):
  u,v=z[:,i],z[:,j];
  for a in (.3,.5,1,2):add(u*v/(1+a*(u*u+v*v)),f'ratprod {i},{j},{a}')
  for c in (-2,-1,-.5,.5,1,2):
   h=u+c*v
   for a in (.5,1,1.5,2):
    for fn in (np.sin,np.cos,np.tanh):add(fn(a*h),f'{fn.__name__}({a}*({i}+{c}*{j}))')
 return np.column_stack(f),names
out=[]
for k,z in Z.items():
 for a in (.7,1.3,2.2):
  for tag,F in [('q',des(z)[:,1:15]),('c',des(z)[:,1:]),('qt',np.c_[des(z)[:,1:15],des(np.tanh(a*z))[:,1:15]]),('ct',np.c_[des(z)[:,1:],des(np.tanh(a*z))[:,1:15]])]:
   for reg in (.003,.03,.3,1):
    sc=ridge(F,reg); out.append((max(sc[0]),max(sc.max(0)),k,tag,a,reg,sc[0]))
 F,names=lib(z)
 for k2 in (4,6,9,12,16,22):
  er=[]
  for fi,(i,j) in enumerate(splits):
   A=F[i]-F[i].mean(0); sd=np.maximum(np.linalg.norm(A,axis=0),1e-9); A=A/sd; D=(F[j]-F[i].mean(0))/sd; pred=[]
   for o in range(3):
    y=targets[i,o];r=y-y.mean(); ids=[]
    for _ in range(k2):
     c=abs(A.T@r); c[ids]=0; ids.append(np.argmax(c)); coef=fit(A[:,ids],y-y.mean());r=y-y.mean()-A[:,ids]@coef
    pred.append(D[:,ids]@coef+y.mean());
    if k=='z' and k2==9 and fi==0:print('OMP',o,np.std(r),[names[h] for h in ids]);
   er.append(score(np.column_stack(pred),j))
  sc=np.array(er);out.append((max(sc[0]),sc.max(),k,'omp',k2,0,sc[0]))
print('public',*sorted(out,key=lambda r:r[0])[:12],sep='\n'); print('worst',*sorted(out,key=lambda r:r[1])[:10],sep='\n')
