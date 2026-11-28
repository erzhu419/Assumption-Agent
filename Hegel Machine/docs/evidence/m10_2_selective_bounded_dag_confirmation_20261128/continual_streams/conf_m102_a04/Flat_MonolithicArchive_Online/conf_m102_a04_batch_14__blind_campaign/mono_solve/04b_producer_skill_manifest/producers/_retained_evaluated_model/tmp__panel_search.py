import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def one(u):return np.c_[np.ones(len(u)),u]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def terms(u):
 p=np.stack([u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)],1);return p
funs={'sin':lambda u,q:np.sin(q*u),'tanh':lambda u,q:np.tanh(q*u),'rat':lambda u,q:u/(1+(q*u)**2),'soft':lambda u,q:u/(1+q*abs(u)),'cos':lambda u,q:np.cos(q*u),'sat2':lambda u,q:u*u/(1+q*u*u),'exp2':lambda u,q:np.exp(-q*u*u)}
p=s['panel_a'];mask=np.linalg.norm(p['v'],axis=1)<4; u,v=p['u'][mask],p['v'][mask]
a=[]
for k,l in itertools.product(['sin','tanh','rat','soft'],repeat=2):
 for j in [1,2,3]:
  def fit(q):
   F=np.stack((np.ones_like(u),u,funs[k](u,q[0]),funs[l](np.roll(u,j,axis=1),q[1])),axis=-1);W=np.stack([ls(F[:,i],v[:,i]) for i in range(4)]);return (F*W).sum(-1)-v
  q=least_squares(lambda q:fit(q).ravel(),[1.4,1.4],bounds=(.05,4),gtol=1e-11).x;a.append((np.std(fit(q)),k,l,j,q))
print('A',sorted(a,key=lambda x:x[0])[:8]);
p=s['panel_c'];u=p['u'];pr=terms(u)
for key in ['v1','v2']:
 v=p[key];results=[]
 for k,fun in funs.items():
  for base,b in [('none',u[:,:0]),('u',u),('p',pr),('u+p',np.c_[u,pr]),('u+sq',np.c_[u,u*u]),('u+sq+p',np.c_[u,u*u,pr]),('sq',u*u)]:
   def Q(q):return one(np.c_[b,fun(u,q[0])]);
   def fit(q):
    a=Q(q);return a@ls(a,v)-v
   for st in [0.3,1.3,2.5]:
    q=least_squares(lambda q:fit(q).ravel(),[st],bounds=(.03,6),gtol=1e-11).x; a=Q(q); e=fit(q); h=np.sum(np.linalg.svd(a,full_matrices=False)[0]**2,1);results.append((np.std(e/np.maximum(1-h[:,None],.001)),np.std(e),k,base,q[0]))
 print(key,sorted(results)[:15])
