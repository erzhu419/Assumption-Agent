import importlib.util,pickle,numpy as np,sys
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; p=s['panel_c'];u=p['u'];v=p['v1']; N=len(tr['x']); Y=np.c_[tr['y'],tr['traj']]; a,b=m._align(s['panel_a']); X=(tr['x']-a)/b

def soft(x,k):return x/(1+k*abs(x))
def cfit(I,mode):
 def F(x,q):
  if mode=='tied':return m.one(soft(x,q[0])+q[1]*x)
  if mode=='untied':return np.c_[m.one(x),soft(x,q[0])]
  return m.one(soft(x,q[0]))
 def r(q):
  f=F(u[I],q);return (f@m.lin(f,v[I])-v[I]).ravel()
 q=least_squares(r,[.85,-.01] if mode=='tied' else [.85],bounds=([.02,-1],[10,1]) if mode=='tied' else (.02,10)).x
 B=m.lin(F(u[I],q),v[I]);return lambda x:F(x,q)@B,q
I=np.arange(len(u)); pf=np.array_split(np.random.default_rng(42).permutation(I),4)
C={}
for mo in ['tied','untied','pure']:
 f,q=cfit(I,mo);err=np.vstack([cfit(np.setdiff1d(I,j),mo)[0](u[j])-v[j] for j in pf]);print(mo,q,'panel fit/OOF RMSE',np.std(f(u)-v),np.std(err));C[mo]=f(X)
for k,f in enumerate(m.fitc(p)):C['old'+str(k)]=f(X)
v2=p['v2']; mode=sys.argv[1]
def f2(w,k):
    q=w*w
    return np.c_[m._square(w),q/(1+k*q)]
def r2(q):
    f=f2(u,q[0]);return (f@m.lin(f,v2)-v2).ravel()
k=least_squares(r2,[.25],bounds=(.08,.65),gtol=1e-13).x[0]
f=f2(u,k); beta=m.lin(f,v2)
print('v2 fit',k,np.std(f@beta-v2),'coefficients',beta.round(4),flush=True)
d0=m._square(X)@m.lin(m._square(u),v2);d1=f2(X,k)@beta
D={'rational':d1,'blend':.5*(d1+d0)}[mode]

A=np.arange(N); rad=np.linalg.norm(X,axis=1);i52=A[:52]
inner=[(i52,np.array_split(np.random.default_rng(j).permutation(i52),4)) for j in (44,4,15,62)]
outer=[('public',A[52:]),('radial_high',np.argsort(rad)[-20:]),('radial_low',np.argsort(rad)[:20]),('early',A[:20]),('mid',A[20:40])]+[(f'r{j}',np.random.default_rng(j).permutation(A)[:20]) for j in range(10)]
def project(i,S):
 def F(r):return -np.expm1(-t[:,None]*r)
 def residual(r):
  f=F(r); return (f@m.lin(f,Y[i,1:].T)-Y[i,1:].T).ravel()
 f=F(least_squares(residual,[.4,1.8],bounds=([.05,.8],[.8,4])).x);return np.c_[S[:,0],(S[:,1:]@np.linalg.pinv(f).T)@f.T]
def predict(i,j,models,kernels):
 out=[]
 for mo in models:
  w=np.c_[C[mo],D];w=(w-w[i].mean(0))/np.maximum(w[i].std(0),.1);G=w@w.T/w.shape[1];d=G.diagonal();d=np.maximum(d[:,None]+d-2*G,0)
  dic={'quad':(1+.25*G)**2,'mix':(1+.25*G)**2+np.exp(-.06*d),'cub':(1+.06*G)**3}
  for k,h in kernels:out.append(dic[k][np.ix_(j,i)]@np.linalg.solve(dic[k][np.ix_(i,i)]+h*np.eye(len(i)),Y[i]))
 return np.mean(out,axis=0)
def score(e):
 e=abs(e).mean(0);return np.array([e[0],e[1:].mean(),e[1:].max()])
rows=[]
choices=[('quad',3e-5),('quad',3e-4),('mix',3e-5),('mix',3e-4),('cub',3e-5)]
for models in [('old0','old2'),('tied',),('untied',),('pure',),('tied','old0')]:
 for kernels in [[q] for q in choices]+[[choices[3],choices[4]],[choices[0],choices[3]]]:
  for proj in [False,True]:
   def err(i,j):
    H=predict(i,j,models,kernels);return (project(i,H) if proj else H)-Y[j]
   em=np.array([score(np.vstack([err(np.setdiff1d(pool,j),j) for j in js])) for pool,js in inner]); eo=np.array([score(err(np.setdiff1d(A,j),j)) for _,j in outer]);
   rows.append((np.max(em[:,[0,2]]),models,kernels,proj,em.max(0),eo[0],eo.max(0)))
print('ranked: inner max, models, kernels, time projection; inner/public/stress y, mean trajectory, worst horizon'); print(*sorted(rows,key=lambda r:r[0])[:12],sep='\n');print('baseline',*[r for r in rows if r[1]==('old0','old2') and r[2]==[choices[3],choices[4]]],sep='\n')
