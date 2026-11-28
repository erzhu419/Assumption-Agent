exec(open('/tmp/probe.py').read().split('Q={};')[0])
from scipy.spatial.distance import cdist
node=load('/holdout_task/node_01.py')
c=node._calibrated(x,s['panel_a']); p=s['panel_b']; b=affine(c)@np.linalg.lstsq(affine(p['u']),p['v'],rcond=None)[0]
f=node._basis(b/np.sqrt(1+.3*b*b),3); K0=f@f.T
f=node._basis(c,2); trend=f@f.T
G=lambda u,l:np.exp(-cdist(u,u,'sqeuclidean')/(2*l*l))
K=trend+(G(x,1.3)+G(c,1.3))/2
Kadd=trend+sum(G(c[:,j:j+1],.65) for j in range(c.shape[1]))
Krot=trend+sum(G(b[:,j:j+1],.65) for j in range(b.shape[1]))
Kall=K+.25*(Kadd+Krot-2*trend)
def pred(K,tr,te,lam):return K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),y[tr],assume_a='pos')
a=np.arange(n); rng=np.random.default_rng(20); old=[(p[:52],p[52:]) for p in [a]+[rng.permutation(n) for _ in range(14)]]
rng=np.random.default_rng(821); fresh=[(p[:52],p[52:]) for p in [rng.permutation(n) for _ in range(60)]]
blocked=[]
for u in [x,c,b]:
 for j in range(u.shape[1]):
  p=np.argsort(u[:,j]); blocked.extend([(p[:58],p[58:]),(p[14:],p[:14])])
res=[]
for tr,te in old+fresh+blocked:
 p0=pred(K0,tr,te,.01); pk=pred(K,tr,te,.001); pa=pred(Kall,tr,te,.001)
 res.append([np.mean(abs(v-y[te])) for v in [p0,pk,.25*p0+.75*pk,.5*p0+.5*pk,pa,.25*p0+.75*pa,pred(Kadd,tr,te,.001),pred(Krot,tr,te,.001)]])
e=np.array(res)
for name,ix in [('official',slice(0,1)),('old',slice(0,len(old))),('fresh',slice(len(old),len(old)+len(fresh))),('blocked',slice(len(old)+len(fresh),None))]:
 print(name,'mean',e[ix].mean(0),'p90',np.quantile(e[ix],.9,axis=0),'max',e[ix].max(0))
