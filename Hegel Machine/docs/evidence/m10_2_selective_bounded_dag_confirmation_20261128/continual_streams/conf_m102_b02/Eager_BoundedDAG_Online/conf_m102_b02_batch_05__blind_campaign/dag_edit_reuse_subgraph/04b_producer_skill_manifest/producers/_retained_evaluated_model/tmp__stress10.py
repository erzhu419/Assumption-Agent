exec(open('/tmp/probe9.py').read().split('base=m._fold_b')[0]);
zall=m._invert(np.vstack([s[k]['x'] for k in ['train','test']]),m._fit_a(s['panel_a']));n0=len(folds);tr=np.arange(n-20);te=np.arange(n-20,n);folds.append((tr,te,f1['run_node']({'__input__':{'train':{'y':truth[tr,0],'traj':truth[tr,1:]}}})))
def metrics(z,b):
 out=[]
 for tr,te,p in folds:
  o=f2['run_node']({'node_00':dict(z_train=z[tr],z_test=z[te],b_train=b[tr],b_test=b[te]),'node_01':p}); e=np.abs(np.c_[o['y_hat'],o['state_pred']]-truth[te]);out.append([e[:,0].mean(),e[:,1:].mean()]);
 a=np.array(out); return a[-1],a[:-1].mean(0),a[:-1].reshape(5,5,2).mean(1).max(0)
def white(x):
 _,s,w=np.linalg.svd(x-x.mean(0),full_matrices=False);return (x-x.mean(0))@((w.T/s)@w)
def lift(x,degree):
 return np.column_stack([np.prod(x[:,i],axis=1) for i in itertools.combinations_with_replacement(range(x.shape[1]),degree)])
fns={'erf':lambda a,q:erf(q[0]*a)+q[1]*a,'exp':lambda a,q:a*np.exp(-(q[0]*a)**2)+q[1]*a,'poly5':lambda a,q:a+q[0]*a**3+q[1]*a**5,'cubic':lambda a,q:a+q[0]*a**3,'rational':lambda a,q:a*(1+q[1]*a*a)/(1+q[0]*a*a+q[2]*a**4)}
res=[]
for nm,fn in fns.items():
 init={'erf':[[1.5,-1.55]],'exp':[[1,-.9]],'poly5':[[-12,3],[-2,1]],'cubic':[[-2,1]],'rational':[[1,-1,1],[0,-2,4],[0,-5,0]]}[nm];
 def design(a,q):return np.c_[np.ones(len(a)),fn(a,q)]
 def fit(q):
  d=design(u,q); w=np.linalg.lstsq(d,v,rcond=None)[0];return w,(d@w-v).ravel()
 sol=min([least_squares(lambda q:fit(q)[1],i,max_nfev=500) for i in init],key=lambda r:r.cost); q=sol.x;w,r=fit(q);F=fn(zall,q);B=design(zall,q)@w; print(nm,q,np.std(r),flush=True)
 for bn,b in [('mixed',B),('source',F),('white',white(B)),('white-src',white(F)),('both',np.c_[F,B])]:
  for zn,za in [('raw',zall),('white',white(zall)),('diag2',np.c_[zall,zall*zall]),('diag3',np.c_[zall,zall**3]),('full2',np.c_[zall,lift(zall,2)]),('zfold',np.c_[zall,F])]:
   for ext,bx in [('none',b),('linear',np.c_[b,zall]),('z2',np.c_[b,zall*zall]),('b2',np.c_[b,b*b]),('mixdiag',np.c_[b,zall*F]),('mixall',np.c_[b,(zall[:,:,None]*F[:,None,:]).reshape(len(zall),-1)]),('zs_b',np.c_[b,b,zall]),('z_quad',np.c_[b,lift(zall,2)])]:
    pub,avg,worst=metrics(za,bx);res.append((max(pub),max(worst),nm,bn,zn,ext,pub,avg,worst))
print('PUBLIC BEST');
for row in sorted(res,key=lambda x:x[0])[:28]:print(row)
print('JOINT ROBUST BEST');
for row in sorted(res,key=lambda x:max(x[:2]))[:24]:print(row)
print('UNCENTERED B BASES');
for row in res:
 if row[4:6]==('raw','none'): print(row)
