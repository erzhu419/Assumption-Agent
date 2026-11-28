import sys,pickle, numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_b02_batch_05_solver as old
from scipy.special import erf
from scipy.optimize import least_squares
D=old.D; fit=old.fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z,b0=old.parts(s); z=z[:len(x)]; b0=b0[:len(x)]
a=s['panel_a']; _,_,e=old.cal(s,x); f=old.AF(a['u']); w=fit(f[e<2],a['v'][e<2]); print('A residual',np.quantile(np.linalg.norm(f@w-a['v'],axis=1),[0,.5,.9,1]),'coefficients',w.round(3))
fns={'aff':(lambda x,a,b:x,[1,0]), 'tanh':(lambda x,a,b:np.tanh(a*x)+b*x,[1,0]), 'atan':(lambda x,a,b:np.arctan(a*x)+b*x,[1,0]), 'erf':(lambda x,a,b:erf(a*x)+b*x,[1,0]), 'asinh':(lambda x,a,b:np.arcsinh(a*x)+b*x,[1,0]), 'sin':(lambda x,a,b:np.sin(a*x)+b*x,[1,0]), 'soft':(lambda x,a,b:x/(1+a*x*x)**b,[1,.8]), 'softsign':(lambda x,a,b:x/(1+a*abs(x))**b,[1,1]), 'dead':(lambda x,a,b:np.sign(x)*np.maximum(abs(x)-a,0)**b,[.3,1.5]),'clip':(lambda x,a,b:np.clip(x,-a,a)+b*x,[.7,0]), 'exp':(lambda x,a,b:x*np.exp(-a*x*x)+b*x,[1,0]),'pow':(lambda x,a,b:x+a*np.sign(x)*abs(x)**b,[.5,3])}

def panel(u,v,z):
 res=[]
 for name,(fn,p) in fns.items():
  def fe(x,q):return D(fn(x,*q))
  def err(q):
   f=fe(u,q); return (f@fit(f,v)-v).ravel()
  lo,hi=([.01,.05],[20,8]) if name in ('soft','softsign','dead','pow') else ([.01,-10],[20,10]);
  if name in ('dead','clip'):hi[0]=1.45
  qs=[least_squares(err,[aa,p[1]],bounds=(lo,hi),max_nfev=100).x for aa in (p[0],.2 if name in ('dead','clip') else 2.15)]
  q=min(qs,key=lambda p:np.linalg.norm(err(p))); res.append((np.sqrt(np.mean(err(q)**2)),name,q.round(4), fe(z,q)@fit(fe(u,q),v)))
 return sorted(res,key=lambda r:r[0])

coords={'x':x,'z':z,'oldB':b0}; p=s['panel_b']; c=s['panel_c'];
for name,u,t,zz in [('B',p['u'],p['v'],z),('C1',c['u'],c['v1'],z),('C2',c['u'],c['v2'],z),('C12',c['v1'],c['v2'],None)]:
 r=panel(u,t,z if zz is None else zz); print(name,[(round(v,5),k,q.tolist()) for v,k,q,_ in r]);
 if zz is not None:
  for v,k,q,w in r[:4]:coords[name+k]=w
  if name=='C1':c1=r[0][-1]
 if name=='C12':coords[name]=panel(u,t,c1)[0][-1]

if __name__=='__main__':
 from scipy.spatial.distance import cdist
 folds=[(np.arange(52),np.arange(52,72))]; ids=np.random.default_rng(700).permutation(72)
 for val in np.array_split(ids,4):folds.append((np.setdiff1d(ids,val),val))
 def scores(K):
  return np.array([np.mean(abs(y[j]-K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+1e-4*np.eye(len(i)),y[i]))) for i,j in folds])
 all=[]
 for key,v in coords.items():
  for scale in [.3,1.,2.15]:
   for name,ff in [('raw',lambda v:v),('asinh',np.arcsinh),('tanh',np.tanh),('soft',lambda v:v/np.sqrt(1+v*v))]:
    h=ff(scale*v)
    for form,g in [('base',h),('sep',np.c_[.3*z,h]),('sym',np.c_[.3*z+h,.3*z-h]),('mix',np.c_[h,v]),('nonsep',np.arcsinh(np.c_[.3*z+h,.3*z-h]))]:
     dot=g@g.T/g.shape[1]; ds=cdist(g,g,'sqeuclidean')/g.shape[1];
     for degree,decay in [(1,0),(2,0),(3,0),(1,1.4),(2,1.4),(3,.15),(3,1.4)]:
      K=(1+dot)**degree+ (np.exp(-decay*ds) if decay else 0); sc=scores(K); all.append((sc.mean()+.3*sc.max(),*sc.round(4),key,scale,name,form,degree,decay))
 print('Best balanced'); print(*sorted(all)[:16],sep='\n');print('Best public');print(*sorted(all,key=lambda t:t[1])[:10],sep='\n')
