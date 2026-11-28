import runpy,pickle,numpy as np
from itertools import product
m=runpy.run_path('/holdout_task/node_00.py'); sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]; n=len(Y); x,b,c,d=[a[:n] for a in m['_features'](s)]; ids=np.arange(n)
def pair(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
F=np.c_[b,c,d,pair(b,c),pair(c,d)]; scalar=np.r_[np.arange(28),28+np.arange(0,16,5)]; G=F[:,scalar]
vals=[ids[-20:]]+list(np.array_split(np.random.default_rng(63).permutation(n),4))+[np.argsort(np.linalg.norm(x,axis=1))[-15:]]+list(np.array_split(np.random.default_rng(5521).permutation(n),4))+[ids[:20]]
def fit(a,y,tr,reg):
 u=a[tr].mean(0); z=a[tr].std(0); A=(a-u)/z; q,sv,v=np.linalg.svd(A[tr],full_matrices=False)
 C=(v.T*(sv/(sv**2+reg)))@(q.T@(y[tr]-y[tr].mean(0)))
 return A@C,C/z[:,None]
def embed(v,tr,rs,rt,k,scale):
 a,_=fit(G,Y[:,:1],tr,rs); p,_=fit(F,Y[:,1:],tr,rt); _,ss,V=np.linalg.svd(Y[tr,1:]-Y[tr,1:].mean(0),full_matrices=False); h=np.c_[a,p@V[:k].T]; h=(h-h[tr].mean(0))/h[tr].std(0)*scale
 o=sink({'node_00':{'h':h[np.r_[tr,v]],'n_train':len(tr),'y':t['y'][tr],'traj':t['traj'][tr]}}); e=np.abs(np.c_[o['y_hat'],o['state_pred']]-Y[v]); return [e[:,0].mean(),e[:,1:].mean()]
order=[]
for rs,rt,k,scale in product((1e-6,.002,.02),(1e-6,.0002,.002,.02,.1),(2,3,4),(.25,.5,1.)):
 E=np.array([embed(v,np.setdiff1d(ids,v),rs,rt,k,scale) for v in vals]); order.append((E.max(),E.mean(),(rs,rt,k,scale),E))
for _,avg,cfg,E in sorted(order,key=lambda a:a[0])[:14]: print(cfg, 'public',E[0].round(5),'worst',E.max(0).round(5),'high radius',E[5].round(5),'avg',round(avg,5))
P=np.linalg.svd(Y[:,1:]-Y[:,1:].mean(0),full_matrices=False)[2][:3].T
_,C=fit(F,np.c_[Y[:,0],Y[:,1:]@P],ids,.002)
for k in range(4):
 print('mode',k,'BC singular',np.linalg.svd(C[12:28,k].reshape(4,4),compute_uv=False).round(4),'CD singular',np.linalg.svd(C[28:44,k].reshape(4,4),compute_uv=False).round(4));print('CD matrix',C[28:44,k].reshape(4,4).round(3))
