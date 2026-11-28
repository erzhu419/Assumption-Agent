exec(open('/tmp/explore.py').read().split('ops={}')[0])
from scipy.optimize import least_squares
p=s['panel_b']; B=lstsq(aff(p['u']),p['v'])[0]; w=aff(z)@B
rng=np.random.default_rng(912); splits=[(np.arange(52),np.arange(52,72))]+[(i[:52],i[52:]) for i in [rng.permutation(72) for k in range(5)]]
def vp(D,y,i,j):
 b=lstsq(D[i],y[i])[0]; return np.mean(abs(D[i]@b-y[i])),np.mean(abs(D[j]@b-y[j])),b
L=np.c_[aff(w),np.sum(w[:,:2]*w[:,2:],axis=1)]
print('traj B coef',lstsq(aff(w),t)[0]); print('base scalar',vp(L,y,*splits[0]));
for perm in [(0,2,1,3),(0,3,1,2),(0,1,2,3)]:
 a,b,c,d=w[:,perm].T; L=np.c_[aff(w),a*b,c*d]
 for tag in ['cube','ratcube','satdiff','tanhdiff','mul2','mul2rat','qrat','qmod','sin']:
  def design(q,tag):
   v=w@q[:4]+q[4]; g=q[5]; h=w@q[6:10]+q[10];
   if tag=='cube': f=v**3
   if tag=='ratcube': f=v**3/(1+g*g*v*v)
   if tag=='satdiff': f=v/(1+g*g*v*v)
   if tag=='tanhdiff': f=np.tanh(g*v)
   if tag=='sin':f=np.sin(g*v)
   if tag=='mul2': f=h*v*v
   if tag=='mul2rat': f=h*v*v/(1+g*g*v*v)
   if tag=='qrat': return L/(1+g*g*v[:,None]**2)
   if tag=='qmod': return np.c_[L,L[:,1:5]*v[:,None]**2/(1+g*g*v[:,None]**2)]
   return np.c_[L,f]
  def search(i,j,n):
   def fun(q):
    D=design(q,tag)[i]; return D@lstsq(D,y[i])[0]-y[i]
   pp=[]
   for k in range(n):
    q=rng.normal(size=11)*.55; q[5]=.469; r=least_squares(fun,q,max_nfev=120);pp.append((np.linalg.norm(r.fun),r.x))
   q=min(pp,key=lambda a:a[0])[1]; D=design(q,tag); return vp(D,y,i,j),q
  (tr,va,coef),q=search(*splits[0],13)
  if va<.037 or tr<.017:
   vals=[search(i,j,7)[0][1] for i,j in splits[1:]]; print(perm,tag,'tr va',tr,va,'repeat',np.round(vals,5),'params',q,'coef',coef,flush=True)
