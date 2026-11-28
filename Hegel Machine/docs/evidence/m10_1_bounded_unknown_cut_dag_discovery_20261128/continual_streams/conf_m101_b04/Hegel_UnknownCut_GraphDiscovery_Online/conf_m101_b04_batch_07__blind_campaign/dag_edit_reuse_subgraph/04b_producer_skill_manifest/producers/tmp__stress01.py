exec(open('/tmp/probe_node01.py').read().split("print('ranges'")[0])
from scipy.linalg import solve
from scipy.spatial.distance import cdist
import importlib.util
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/node_01.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
b=mod._panel_coordinates(s['panel_b'],x)
folds=[*np.array_split(np.arange(len(x)),4),np.argsort(np.max(abs(x),1))[-16:],np.argsort(np.linalg.norm(b,axis=1))[-16:]]
for te in folds:
 tr=np.setdiff1d(np.arange(len(x)),te); A=np.c_[np.ones(len(x)),x,b]; yp=mod._scalar(b[tr],b[te],y[tr]); zp=A[te]@np.linalg.lstsq(A[tr],z[tr],rcond=None)[0]; print('baseline',np.mean(abs(yp-y[te])),np.mean(abs(zp-z[te])),np.mean(abs(zp-z[te]),0))

def kern(w, p):
 if p=='rbf': return np.exp(-.5*cdist(w,w,'sqeuclidean'))
 if p=='add': return np.exp(-.5*(w[:,None]-w[None,:])**2).mean(axis=2)
 return (w@w.T)**int(p)
def predict(tr,te,case):
 bl,ke,l,quad,add,reg,d4=case
 bx=b / np.std(b[tr],0)/2; xx=x / np.std(x[tr],0)/2
 A=np.c_[np.ones(len(x)),b];
 if bl: A=np.c_[A,x]
 K=kern(bx/l,ke)+quad*kern(bx,'2')+add*kern(xx,'add')+d4*(xx[:,0,None]*xx[:,0])**4
 G=K[np.ix_(tr,tr)]+reg*np.eye(len(tr)); D=A[tr]; P=solve(G,np.c_[y[tr],D],assume_a='pos'); c=np.linalg.solve(D.T@P[:,1:],D.T@P[:,0]); return A[te]@c+K[np.ix_(te,tr)]@(P[:,0]-P[:,1:]@c)
choices=list(itertools.product([False,True],['rbf','add'],[.7,1.,1.5,2.],[0,.1,1.],[0,.1,1.],[.005,.02,.05],[0.,.25,1.]))
res=[]
for case in choices:
 m=np.array([np.mean(abs(predict(np.setdiff1d(np.arange(len(y)),te),te,case)-y[te])) for te in folds]);res.append((max(m),m.mean(),case,m))
for q in sorted(res,key=lambda q:q[:2])[:12]: print('challenger',q)
