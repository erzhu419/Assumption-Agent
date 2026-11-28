import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from node_01 import calibrate,poly
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=calibrate(s['panel_a'],s['train']['x']); y=np.c_[s['train']['y'],s['train']['traj']]; n=len(y)
fit=lambda X,Y:np.linalg.lstsq(X,Y,rcond=1e-10)[0]
base=np.ones(n); rng=np.random.default_rng(893); splits=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),v),v) for v in np.array_split(rng.permutation(n),4)]
fun={'sin':np.sin,'tanh':np.tanh,'r':lambda x:x/(1+x*x),'r2':lambda x:x*x/(1+x*x),'cos':np.cos}
F={'z':z,'x':s['train']['x']}; maps={}
for key,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[key]['u'],s[key][j]; L=[]
 for name,f in fun.items():
  def get(p,x):return np.c_[np.ones(len(x)),x,f(p*x)]
  def cost(p):
   a=get(p,u); q=np.linalg.qr(a)[0];return np.mean(((v-a@fit(a,v))/(1-(q*q).sum(1))[:,None])**2)**.5
  res=min([minimize_scalar(cost,bounds=b,method='bounded') for b in ((.12,.6),(.6,1.4),(1.4,2.5),(2.5,4),(4,6))],key=lambda r:r.fun); L.append((res.fun,name,res.x))
 e,name,p=min(L); a=get(p,u); B=fit(a,v); print(j,sorted(L), 'linear',B[1:5], 'nonlinear', B[5:],flush=True); maps[j]=(name,p,B); F[j]=get(p,z)@B
 # Estimate the unmixed coordinate-wise perturbation using the parallel coefficients.
 m=(B[1:5]*B[5:]).sum(1)/(B[1:5]**2).sum(1); F['u'+j]=z+fun[name](p*z)*m
np.savez('/tmp/search_maps.npz',**F)
results=[[],[]]
def trial(name,A,regs=(0.,.001,.03,.2)):
 for reg in regs:
  errors=[]
  for tr,va in splits:
   mu=A[tr].mean(0); sd=np.maximum(A[tr].std(0),1e-8); D=(A-mu)/sd; m=y[tr].mean(0); V=y[tr]-m
   if reg:
    K=D[tr]@D[tr].T; coef=D[tr].T@np.linalg.solve(K+reg*np.eye(len(K)),V)
   else:coef=fit(D[tr],V)
   e=abs(D[va]@coef+m-y[va]); errors.append((e[:,0].mean(),e[:,1:].mean()))
  E=np.array(errors)
  for j in range(2):results[j].append((E[:,j].mean()*.55+E[:,j].max()*.45,name,reg,np.round(E[:,j],4)))
for key,w in F.items():
 for key2,v in F.items():
  if key>=key2:continue
  trial('fused '+key+' '+key2,np.c_[poly(w,2),poly(v,2)])
  trial('full '+key+' '+key2,poly(np.c_[w,v],2))
for name in ('sin','tanh','r'):
 f=fun[name]
 for g in (.4,.7,1.,1.2,1.5,1.8,2.1,2.3,2.5,2.7,2.8,3.,3.2,3.5,4.):
  w=f(g*z); tag=f'{name} {g}'; trial('full sep '+tag,poly(np.c_[z,w],2)); trial('diag sep '+tag,np.c_[poly(z,2),w,z*w,w*w]);
  for m in (-2,-1,-.4,.4,1,2):trial(f'warp {tag} {m}',poly(z+m*w,2),(0.,))
  trial('sum '+tag,np.c_[poly(z,2),poly(w,2)])
  trial('cross '+tag,np.c_[poly(z,2),w,(z[:,:,None]*w[:,None,:]).reshape(n,16)])
for i in range(2):
 print('BEST',i,*sorted(results[i],key=lambda x:x[0])[:18],sep='\n',flush=True)
# Optimize a compact structured readout with the same shape library; fit shape parameters inside every split.
for nm in ('sin','tanh','r'):
 fn=fun[nm]
 for form in ('warped','interaction'):
  for target in (0,1):
   V=y[:,0:1] if not target else y[:,1:]; er=[]; pars=[]
   for tr,va in splits:
    best=None
    for g0 in (.8,1.7,2.8,3.8):
     def design(g,x):
      w=fn(x*g[0]);
      if form=='warped':return poly(x+w*g[1:],2)
      return np.c_[poly(x,2),w,(x[:,:,None]*w[:,None,:]).reshape(len(x),16),w*w]
     def res(g):
      A=design(g,z[tr]); return (A@fit(A,V[tr])-V[tr]).ravel()
     init=np.r_[g0,np.repeat(-.4,4)] if form=='warped' else np.array([g0]); r=least_squares(res,init,bounds=(np.r_[.2,np.repeat(-8,len(init)-1)],np.r_[5.8,np.repeat(8,len(init)-1)]),max_nfev=45); score=np.mean(r.fun**2)
     if best is None or score<best[0]:best=(score,r.x)
    g=best[1]; er.append(np.mean(abs(design(g,z[va])@fit(design(g,z[tr]),V[tr])-V[va])));pars.append(g)
   print('OPT',nm,form,target,np.round(er,4),np.round(pars,4),flush=True)
