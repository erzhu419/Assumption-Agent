import itertools,pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; c=s['panel_c']; u=c['u']; w=f['_affine'](f['_affine'](x)@f['_panel'](s['panel_a'],3))@f['_panel'](s['panel_b']); a=f['_affine'](x)@f['_panel'](s['panel_a'],3)
ix=list(itertools.combinations_with_replacement(range(4),2))
def Q(x): return np.column_stack([np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in ix]])
R=np.random.default_rng(42); splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),v),v) for v in np.array_split(R.permutation(72),4)]
def check(p,Y):
 return np.array([np.abs(p[v]@np.linalg.lstsq(p[t],Y[t],rcond=None)[0]-Y[v]).mean() for t,v in splits]).round(4)
U,S,V=np.linalg.svd(T,full_matrices=False); print('time factors',V[:2].round(4),'amplitude min/max', (T@V[0]).min(),(T@V[0]).max())
for raw in [x,a,w]:
 print('curve test',[(k,check(Q(raw),np.arctanh(np.clip((T@V[0])/k,-.999,.999)))) for k in [1,1.5,2,3,4]])
co=np.linalg.lstsq(Q(u),c['v2'],rcond=None)[0]; print('v2 quad coeffs',co.round(3));
for name,raw in [('x',x),('a',a),('w',w)]:
 P=Q(raw)@co; print('v2 transfer',name,check(np.c_[np.ones(len(raw)),P],T),check(np.c_[np.ones(len(raw)),np.tanh(P)],T));
forms={'linear+sine':np.sin,'linear+tanh':np.tanh,'linear+softsign':lambda t:t/np.sqrt(1+t*t),'linear+cos':np.cos}
A=Q(u)[:,:5]; B=Q(u)
for name,phi in forms.items():
 for linear in [True,False]:
  pars=[]; errors=[]
  for j,Y in enumerate(c['v1'].T):
   base=A if linear else A[:,:1]
   models=[]
   for _ in range(16):
    p0=np.r_[np.linalg.lstsq(base,Y,rcond=None)[0],R.uniform(-.5,.5),R.normal(0,.8,5)]
    def pred(p,b,l):return b@p[:b.shape[1]]+p[-6]*phi(l@p[-5:])
    r=least_squares(lambda p:pred(p,base,A)-Y,p0,max_nfev=350); models.append((np.linalg.norm(r.fun),r.x))
   err,p=min(models,key=lambda x:x[0]); pars.append(p); errors.append(err/np.sqrt(len(u)))
  out=[]
  for raw in [x,a,w]:
   l=Q(raw)[:,:5]; b=l if linear else l[:,:1]; P=np.column_stack([pred(p,b,l) for p in pars]); out.append(check(np.c_[np.ones(len(raw)),P],y))
  print(name,linear,np.round(errors,4),out)
