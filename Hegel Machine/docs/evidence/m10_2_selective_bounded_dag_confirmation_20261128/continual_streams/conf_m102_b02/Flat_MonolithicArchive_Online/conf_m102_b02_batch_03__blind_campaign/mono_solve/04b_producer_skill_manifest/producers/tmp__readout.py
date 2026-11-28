import numpy as np,pickle,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; n=52
z,w=np.load('/tmp/coords.npz').values(); v1=np.load('/tmp/v1.npz');v2=np.load('/tmp/v2.npz'); f={'z':z,'w':w,'a':v1['z1'],'b':v1['w1'],'c':v2['z1'],'d':v2['w1']}; m=np.linalg.svd(T[:n],full_matrices=False)[2][:2]; Y=np.c_[y,T@m.T]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
print('times',t,'summary',tr['summary'],'stats',np.corrcoef(Y.T));print('ranges',[(k,v.min(0),v.max(0)) for k,v in f.items()]); print('moments',y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std())
D=dict(f); lin=list(f)
for k,x in f.items():
 for name,a in [('sq',x*x),('cub',x**3),('cyc',x*np.roll(x,-1,axis=1)),('tanh',np.tanh(1.25*x)),('asinh',np.arcsinh(1.25*x)),('rat',x/(1+.22*x*x)),('quad',np.array([x[:,i]*x[:,j] for i,j in it.combinations_with_replacement(range(4),2)]).T),('sum2',(x*x).sum(1)),('sum3',(x**3).sum(1)),('tanh_sum',np.tanh(1.25*x).sum(1)),('normrat',x/(1+.22*(x*x).sum(1)[:,None]))]:D[k+name]=a
for (k,x),(l,h) in it.combinations(f.items(),2):
 D[k+l+'dot']=np.sum(x*h,axis=1);D[k+l+'diag']=x*h;D[k+l+'outer']=(x[:,:,None]*h[:,None,:]).reshape(len(x),-1)

def measure(keys):
 a=np.column_stack([np.ones(len(z))]+[D[k] for k in keys]);
 if a.shape[1]>36:return None
 a/=np.maximum(np.linalg.norm(a[:n],axis=0),1e-9)
 c=ls(a[:n],Y[:n]);r=a@c; e=Y-r[:len(Y)]; h=(a[:n]*np.linalg.pinv(a[:n]).T).sum(1); loo=e[:n]/np.maximum(1-h[:,None],1e-5)
 mae=lambda q:np.r_[np.mean(abs(q),axis=0),np.mean(abs(q[:,1:]@m))]
 return mae(loo),mae(e[n:]),mae(e[:n]),a.shape[1],keys
base=[(k,) for k in lin]+list(it.combinations(lin,2))+list(it.combinations(lin,3))
R=[]
for b in base:
 R.append(measure(b))
 for k in D:
  if k not in lin:R.append(measure(b+(k,)))
R=[r for r in R if r is not None]
for col in [0,1,2,3]:
 print('BASE',col)
 best=sorted(R,key=lambda r:r[0][col])[:14]
 for a,b,c,p,k in best[:6]:print(k,p,'loo',a.round(4),'gate',b.round(4),'train',c.round(4))
 for j in range(2):
  best=sorted([q for r in best for k in D if k not in r[-1] and (q:=measure(r[-1]+(k,))) is not None],key=lambda r:r[0][col])
  uniq={};
  for r in best:uniq.setdefault(tuple(sorted(r[-1])),r)
  best=list(uniq.values())[:12]
 print('EXPANDED',col)
 for a,b,c,p,k in best[:5]:print(k,p,'loo',a.round(4),'gate',b.round(4),'train',c.round(4))
