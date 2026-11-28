import pickle, numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u,v=a['u'],a['v']
np.set_printoptions(precision=5,suppress=True)
def aff(x): return np.c_[np.ones(len(x)),x]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
mask=np.linalg.norm(v,axis=1)<max(4,3*np.median(np.linalg.norm(v,axis=1)))
print('A norms',np.sort(np.linalg.norm(v,axis=1))); print('times',s['eval_times']);
for roll in range(4):
 def feats(x,p):
  a,e,b,f=p; w=np.roll(x,roll,axis=1); return np.stack((np.ones_like(x),x/np.sqrt(1+a*abs(x)**e),w/(1+b*abs(w)**f)),axis=2)
 def get(p):
  h=feats(u[mask],p); c=np.stack([ls(h[:,j],v[mask,j]) for j in range(4)]); return np.einsum('ijk,jk->ij',h,c)-v[mask],c
 for p in (s['telemetry_vec'],least_squares(lambda p:get(p)[0].ravel(),s['telemetry_vec'],bounds=([.001,.3,.001,.3],[8,5,8,5])).x):
  r,c=get(p);print('A',roll,p,np.sqrt(np.mean(r*r)),c.T)
for k in ['panel_b','panel_c']:
 u=s[k]['u']; print(k)
 for j,v in s[k].items():
  if j=='u':continue
  def fit(x):h=aff(x);return h@ls(h,v)-v
  for n,f,p in [('linear',lambda x,p:x,[]),('tanh',lambda x,p:np.tanh(p[0]*x+p[1]),[1,0]),('sin',lambda x,p:np.sin(p[0]*x+p[1]),[1,0]),('squarewarp',lambda x,p:x+p[0]*x*x,[.4]),('softplus',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2,0]),('rational',lambda x,p:np.sign(x)*abs(x)**p[0]/(1+p[1]*abs(x)**p[2]),[1,.22,1.25]),('cubic',lambda x,p:(x**3+p[0]*x)/np.sqrt(1+p[1]*abs(x)**3),[.5,.22])]:
   if p:
    r=least_squares(lambda p:fit(f(u,p)).ravel(),p,max_nfev=250); p=r.x
   print(j,n,p,np.sqrt(np.mean(fit(f(u,p))**2)))
  for n,h in [('additive',np.c_[u,u*u,u**3]),('quadratic',np.c_[u,*[u[:,i]*u[:,l] for i in range(4) for l in range(i,4)]]),('trig',np.c_[np.sin(u),np.cos(u),u])]: print(j,n,np.sqrt(np.mean(fit(h)**2)))
