import pickle,numpy as np,runpy,itertools
m=runpy.run_path('/holdout_task/node_00.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=m['_coordinates'](s); M=m['_maps'](s,X); P=m['_pairs']; fit=m['_fit']; t=s['train']; nn=len(t['x']); y=np.c_[t['y'],t['traj']]; rng=np.random.default_rng(23); splits=[np.arange(nn*3//4,nn),np.arange(52,nn),*np.array_split(rng.permutation(nn),4)]; print('extents',[(w[:nn].min(0),w[:nn].max(0)) for w in X]);
def score(w, norm=True):
 out=[]
 for te in splits:
  tr=np.setdiff1d(np.arange(nn),te); W=w[:nn]-w[tr].mean(0); W=W/np.maximum(w[tr].std(0),.1) if norm else W; g=W@W.T/W.shape[1]; D=np.maximum(g.diagonal()[:,None]+g.diagonal()-2*g,0); Q=(1+.25*g)**2; E=np.eye(len(tr)); pred=sum(K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+r*E,y[tr]) for K,r in [(Q,3e-5),(Q+np.exp(-.06*D),3e-4)])/2; e=abs(pred-y[te]); out.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.array(out)
res=[]
for i,(z,(c,d)) in enumerate(zip(X,M)):
 lib={'z':z,'C':c,'D':d,'p':P(z),'q':z*z,'R':P(c),'S':P(d),'h':z*z/(1+.25*z*z)}
 for j in range(1,4):
  for keys in itertools.combinations(lib,j):
   w=np.column_stack([lib[k] for k in keys]); a=score(w); res.append((a.max(),i,''.join(keys),np.round(a,4).tolist()))
print(*sorted(res)[:24],sep='\n')
