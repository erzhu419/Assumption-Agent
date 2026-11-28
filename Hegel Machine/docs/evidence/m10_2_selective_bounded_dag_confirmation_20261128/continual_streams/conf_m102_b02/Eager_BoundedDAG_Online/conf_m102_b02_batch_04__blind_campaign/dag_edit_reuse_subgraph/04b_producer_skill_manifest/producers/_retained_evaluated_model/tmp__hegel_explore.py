import pickle, numpy as np, runpy
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; AR='/tmp/hegel_continual_archive/conf_m102_b02_batch_01_solver.py'; arch=runpy.run_path(AR)
D=lambda x:np.c_[np.ones(len(x)),x]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
P=lambda x,j:np.column_stack([np.prod(x[:,inds],axis=1) for inds in combinations_with_replacement(range(x.shape[1]),j)])
x=S['train']['x']; tr=S['train']['traj']; y=S['train']['y']; z,b=arch['_coordinates'](S,x)
a=S['panel_a']; print('A inverse range',z.min(0),z.max(0)); p=S['panel_b'];u,v=p['u'],p['v'];
methods={'affine':lambda u:D(u),'quad':lambda u:D(np.c_[u,u*u]),'cross':lambda u:D(np.c_[u,P(u,2)]),'cubic':lambda u:D(np.c_[u,u*u,u**3]), 'odd-cubic':lambda u:D(np.c_[u,u**3])}
for key,g in [('erf',erf),('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh),('softsign',lambda a:a/np.sqrt(1+a*a))]:
 def project(m, uu, vv, g=g):
  def f(q,z): return D(g(q[0]*z)+q[1]*z)
  def r(q):
   F=f(q,uu);return (F@fit(F,vv)-vv).ravel()
  sol=min([least_squares(r,[q,-q/2],bounds=([.1,-10],[5.,10]),max_nfev=100) for q in [.45,1.3,2.8]],key=lambda o:o.cost)
  F=f(sol.x,uu);return f(sol.x,m)@fit(F,vv),sol.x
 yh, q=project(u,u,v); err=[project(u[[i]],np.delete(u,i,0),np.delete(v,i,0))[0][0]-v[i] for i in range(len(u))];print('B',key,q,'fit',np.sqrt(np.mean((yh-v)**2)),'loo',np.sqrt(np.mean(np.array(err)**2))); methods[key]=lambda zz,q=q,g=g:D(g(q[0]*zz)+q[1]*zz)
cs={}
for key,f in methods.items():
 F=f(u); H=np.linalg.pinv(F); e=(v-F@H@v)/(1-(F*H.T).sum(1))[:,None]; print('B fixed',key,'loo',np.sqrt(np.mean(e**2)));cs[key]=f(z)@fit(F,v)
p=S['panel_c']; c1=D(np.c_[z,P(z,2)])@fit(D(np.c_[p['u'],P(p['u'],2)]),p['v1']);c2=D(np.c_[z,z*z])@fit(D(np.c_[p['u'],p['u']**2]),p['v2']);print('C2 coefficients',fit(D(np.c_[p['u'],p['u']**2]),p['v2']).round(3))
fold=[np.r_[52:72]]+[i for i in np.array_split(np.random.default_rng(1234).permutation(len(x)),4)]
configs=[]
for name,w in list(cs.items())+[('z',z),('c1',c1),('c2',c2)]:
 for group,f in [('w',w),('zw',np.c_[z,w]),('both',np.c_[z,w,c2]),('all',np.c_[z,w,c2,c1]),('trig',np.c_[z,w,np.tanh(z),np.tanh(2.5*w)])]: configs.append((name+' '+group,f))
res=[]
for name, f in configs:
 for deg in (1,2,3):
  E=[]
  for te in fold:
   ii=np.setdiff1d(np.arange(len(x)),te); std=np.maximum(f[ii].std(0),1e-7); w=f/std; K=(1+w@w.T/w.shape[1])**deg
   if deg==3:
    K=K+.3*arch['_kernel'](w,w)
   modes=np.linalg.svd(tr[ii],full_matrices=False)[2][:2]; T=np.c_[y[ii],tr[ii]@modes.T]
   row=[]
   for l in [.00001,.0003,.004,.03]:
    hh=K[np.ix_(te,ii)]@np.linalg.solve(K[np.ix_(ii,ii)]+l*np.eye(len(ii)),T);row.append([np.abs(hh[:,0]-y[te]).mean(),np.abs(hh[:,1:]@modes-tr[te]).mean()])
   E.append(row)
  for l,e in zip([.00001,.0003,.004,.03],np.array(E).transpose(1,0,2)):
   res.append((max(e.mean(0)),name,deg,l,e[0],e.mean(0),e.max(0)))
for col in [0,1]:
 print('best',col)
 for r in sorted(res,key=lambda r:r[5][col])[:12]:print(r)
print('both',*sorted(res,key=lambda r:r[0])[:10],sep='\n')
