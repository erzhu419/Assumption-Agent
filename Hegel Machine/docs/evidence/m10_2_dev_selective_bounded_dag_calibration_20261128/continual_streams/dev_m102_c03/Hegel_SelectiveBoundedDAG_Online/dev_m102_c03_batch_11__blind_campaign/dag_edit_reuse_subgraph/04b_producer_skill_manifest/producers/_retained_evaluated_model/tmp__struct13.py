import pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task'); import node_01 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; z,c,w=m._calibrated(s,x); print('calibrated sd',z.std(0),c.std(0),w.std(0)); n=len(x); rng=np.random.default_rng(123); folds=[np.arange(n-20,n)]+[p for _ in range(3) for p in np.array_split(rng.permutation(n),4)]; t=s['eval_times'];
features={}
for label,g in [('tanh',np.tanh(1.6*z)),('c1',c),('soft',z/(1+.83*abs(z))),('z',z),('atan',np.arctan(2*z))]:
 a=np.c_[w,g]; features[label+'all']=np.c_[a,m._quad(a)]; features[label+'sep']=np.c_[a,m._quad(w),m._quad(g)]; features[label+'low']=np.c_[a,m._quad(w)]; features[label+'cross']=np.c_[a,m._quad(w),(w[:,:,None]*g[:,None,:]).reshape(n,-1)];
 for h in [features[label+'all']]: features[label+'extra']=np.c_[h,m._quad(z),z]
for name,h in features.items():
 scores=[]
 for reg in [.0002,.002,.02,.1,.5,2]:
  qv=[]
  for va in folds:
   it=np.setdiff1d(np.arange(n),va); q,a=m._modes(t,tr['traj'][it]); order=np.r_[it,va]; p=m._ridge(h[order],np.c_[tr['y'][it],a],len(it),reg); qv.append([np.mean(abs(p[:,0]-tr['y'][va])),np.mean(abs(p[:,1:]@q.T-tr['traj'][va]))]);
  qv=np.asarray(qv); scores.append((np.mean(qv,axis=0),np.max(qv,axis=0),qv[0],reg))
 print(name, sorted(scores,key=lambda q:max(q[1]))[:2])
