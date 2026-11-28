import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task');sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as m,node_01 as fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];y=s['train']['y'];z=m._calibrate(s['panel_a'],s['train']['x']); b,h=m._branch(s['panel_b'],z); n=len(y); one=np.ones(n); q=m._quad
print('B freq',h, 'z range',z.min(0),z.max(0));print('C keys',s['panel_c'].keys())
cs={}
for name in ('panel_b','panel_c'):
 for kind in ('sin','warp','sintanh','rational'):
  cs[name,kind],_=m._branch(s[name],z,kind,True)
  p=s[name];print('norm', name,np.quantile(np.linalg.norm(p['v'],axis=1),[0,.5,1]))
u,v=[s['panel_b'][k] for k in ('u','v')];freq=h[0];coef=m._fit(m._bdesign(u,h,'sin'),v);print('B co',np.round(coef,3));
sets={'z':z,'b':b,'br':cs['panel_b','rational'],'bw':cs['panel_b','warp'],'c':cs['panel_c','sin'],'h':np.sin(freq*z),'w':z-.4*np.sin(freq*z)}
for a in (.2,.3,.5,.6,.8):sets['w'+str(a)]=z-a*np.sin(freq*z)
folds=[np.arange(i,i+18) for i in (0,18,36)]+[np.arange(52,n)]

def score(A):
 out=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te) if te[0]!=36 else np.r_[0:36,54:72]
  p=A.shape[1];pen=[np.r_[np.ones(4),np.full(10,k),np.full(p-18,j),np.full(4,j*3)] for k in (.15,1.,10.) for j in (3.,30.,300.)]
  c=fit._fit(A[tr],y[tr],pen);out.append(np.mean(abs(A[te]@c-y[te])))
 return np.array(out)

def features(a,d,f,k):
 if k=='baseline':return m._blocks(z,b,h)[0]
 A,D,F=(sets[i] for i in (a,d,f)); middle=[one]
 if k in ('quad','cross','cub','full','add','rad'):middle.append(q(D))
 if k in ('cross','cub','full'):middle.append((A[:,:,None]*D[:,None,:]).reshape(n,-1))
 if k in ('cub','full','add'):middle.extend([A**3,D**3])
 if k in ('add','rad'):middle.extend([A**2*D,A*D**2])
 if k=='full':middle.append(np.column_stack([A[:,i]*A[:,j]*A[:,k] for i,j,k in itertools.combinations_with_replacement(range(4),3)]))
 if k=='rad':middle.extend([D/(1+D*D),np.tanh(D)])
 return np.c_[A,q(F),*middle,D]
res=[]
for a,d,f in itertools.product(sets,repeat=3):
 if d not in ('b','h','w','c','z') or (a not in ('z','b','w','h','c') and a!=f):continue
 for k in ('simple','quad','cross','cub','full','add','rad'):
  r=score(features(a,d,f,k));res.append((r[:3].mean(),r.max(),r[-1],a,d,f,k,r))
r=score(features('z','b','z','baseline')); print('baseline',r)
for order in (lambda x:x[0],lambda x:x[1],lambda x:x[2]):
 print('ranking');
 for row in sorted(res,key=order)[:12]:print(row)
