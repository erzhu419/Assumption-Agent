import numpy as np,pickle,importlib.util,sys
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement

def load(p):
 s=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=st['train'];old=load('/tmp/hegel_continual_archive/conf_m102_b02_batch_01_solver.py');tim=load('/holdout_task/_frozen_nodes/node_01.py'); z=tim.run_node({'__input__':st});x=old._calibrate(st['panel_a'],t['x']); print('A residual inliers',[(lambda r:(np.median(abs(r)),np.sum(abs(r)<.001),np.quantile(abs(r),[.7,.8,.9])))(st['panel_a']['v'][:,j]-np.column_stack((np.ones(48),st['panel_a']['u'][:,j]))@np.linalg.lstsq(np.column_stack((np.ones(48),st['panel_a']['u'][:,j]))[np.abs(st['panel_a']['v'][:,j]-st['panel_a']['u'][:,j])<.5],st['panel_a']['v'][:,j][np.abs(st['panel_a']['v'][:,j]-st['panel_a']['u'][:,j])<.5],rcond=None)[0]) for j in range(4)])

def poly(q,n): return np.column_stack([np.prod(q[:,i],axis=1) for k in range(1,n+1) for i in combinations_with_replacement(range(4),k)])
def fit(p,k,f):
 u,v=p['u'],p[k]; d=np.c_[np.ones(len(u)),f(u)];coef=np.linalg.lstsq(d,v,rcond=None)[0];hat=d@coef;r=v-hat; h=np.sum(np.linalg.svd(d,full_matrices=False)[0]**2,axis=1);return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/np.maximum(.00001,1-h[:,None]))**2)),lambda q:np.c_[np.ones(len(q)),f(q)]@coef
families={'quadratic':lambda q:poly(q,2),'diag2':lambda q:np.c_[q,q*q],'diag3':lambda q:np.c_[q,q*q,q**3]}
even={'exp':lambda q,k:-np.expm1(-k*q*q),'rat':lambda q,k:q*q/(1+k*q*q),'root':lambda q,k:np.sqrt(1+k*q*q)-1,'cos':lambda q,k:1-np.cos(np.sqrt(k)*q),'log':lambda q,k:np.log1p(k*q*q),'cosh':lambda q,k:np.cosh(np.sqrt(k)*q)-1,'tanh':lambda q,k:np.tanh(k*q*q),'siglog':lambda q,k:np.logaddexp(k*q,-k*q)-np.log(2)}
fn={}
for name,key in [('b','v'),('c','v2')]:
 p=st['panel_'+name]; res=[]
 for lab,f in families.items():res.append((*fit(p,key,f),lab))
 for lab,g in even.items():
  for mode in ('add','pure','tied','cube'):
   def f(q,k,g=g,mode=mode):
    h=g(q,k)
    return {'add':np.c_[q,h],'pure':h,'tied':h+.15*q,'cube':np.c_[q,q**3,h]}[mode]
   opt=minimize_scalar(lambda lk:fit(p,key,lambda q:f(q,np.exp(lk)))[0],bounds=(-6,2.5),method='bounded'); k=np.exp(opt.x)
   res.append((*fit(p,key,lambda q,f=f,k=k:f(q,k)),lab+':'+mode+':'+str(round(k,5))))
 res.sort(key=lambda a:a[0]); print(name,'fits',[(round(e,7),round(l,7),lab) for e,l,_,lab in res][:19]);fn[name]={lab:f for e,l,f,lab in res if lab in families or (e<.13 and l<.2)}
def scores(d):
 ans=[]
 for tr,te in splits:
  s=d[tr].std(0);s[s<1e-10]=1; a=d[tr]/s;b=d[te]/s;pr=b@np.linalg.solve(a.T@a+.003*np.eye(a.shape[1]),a.T@z['target'][tr]);ans.append([np.abs(pr[:,0]-t['y'][te]).mean(), np.abs(pr[:,1:]@z['time'].T-t['traj'][te]).mean()])
 return np.array(ans)
splits=[(np.arange(52),np.arange(52,72))];rng=np.random.default_rng(140);splits += [(p[:52],p[52:]) for p in (rng.permutation(72) for _ in range(5))]
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
r=[]
fn['b']['old']=lambda q:old._operator(st['panel_b'],'v',q,'b')
for bl,bf in fn['b'].items():
 b=bf(x);c1=old._operator(st['panel_c'],'v1',b,'c');
 for cl,cf in fn['c'].items():
  c2=cf(b)
  for family,d in [('x*c1',np.c_[x,c1,cross(x,c1)]),('x*c2',np.c_[x,c2,cross(x,c2)]),('x*c12',np.c_[x,c1,c2,cross(x,c1),cross(x,c2)]),('x*c1+c2',np.c_[x,c1,c2,cross(x,c1)]),('x*c2+c1',np.c_[x,c1,c2,cross(x,c2)]),('x*b',np.c_[x,b,cross(x,b)]),('b*c2',np.c_[b,c2,cross(b,c2)]),('x*c1+diagc2',np.c_[x,c1,c2,cross(x,c1),x*c2]),('x*c2+diagc1',np.c_[x,c1,c2,cross(x,c2),x*c1])]:
   sc=scores(np.c_[np.ones(len(x)),d]); r.append((sc.max(),sc[0].tolist(),bl,cl,family))
for row in sorted(r)[:30]:print(row)
