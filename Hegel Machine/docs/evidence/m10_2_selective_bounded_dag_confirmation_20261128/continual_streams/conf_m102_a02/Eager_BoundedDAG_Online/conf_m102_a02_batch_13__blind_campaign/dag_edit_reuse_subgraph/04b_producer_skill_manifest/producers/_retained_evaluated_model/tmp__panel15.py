import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.column_stack([s['train']['y'],s['train']['traj']]);
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
a=s['panel_a']; u,v=a['u'],a['v'];
def ad(z,i,j):
 p,q=z[:,i],z[:,j]; return np.c_[np.ones(len(z)),p,q,p*q,p**3,q**3,p*p,q*q,p*p*q,p*q*q]
for i in range(4):
 opts=[]
 for j in range(4):
  if i==j:continue
  for n in (5,8,10):
   D=ad(u,i,j)[:,:n]; c=least_squares(lambda c:D@c-v[:,i],fit(D,v[:,i]),loss='cauchy',f_scale=.002,max_nfev=600).x
   r=abs(D@c-v[:,i]); opts.append((np.median(r),j,n,c,np.percentile(r,[50,80,90,95,100])))
 print('cal',i,[(j,n,c.round(4),r.round(5)) for _,j,n,c,r in sorted(opts,key=lambda t:t[0])[:3]])
for pn,k in [('b','v'),('c','v1'),('c','v2')]:
 p=s['panel_'+pn]; U,V=p['u'],p[k]; scores=[]
 for name,f in [('tanh',np.tanh),('erf',erf),('sin',np.sin),('cos',np.cos),('rational',lambda v:1/(1+v*v))]:
  for mode in range(4):
   def des(a,w):
    T=[a,a*a,a*np.roll(a,1,axis=1),a+np.roll(a,1,axis=1)][mode]; return np.c_[np.ones(len(a)),a,f(w*T)]
   def loss(w):
    D=des(U,w); Q=np.linalg.qr(D)[0]; R=(V-Q@(Q.T@V))/(1-(Q*Q).sum(1))[:,None]; return np.sqrt(np.mean(R*R))
   ws=np.linspace(.25,7,80); j=np.argmin([loss(w) for w in ws]); opt=minimize_scalar(loss,bounds=(ws[max(0,j-1)],ws[min(j+1,len(ws)-1)]),method='bounded'); scores.append((opt.fun,name,mode,opt.x))
 print('panel',pn,k,'n',len(U),'best',sorted(scores)[:8]);
 for _,name,mode,w in sorted(scores)[:1]:
  f={'tanh':np.tanh,'erf':erf,'sin':np.sin,'cos':np.cos,'rational':lambda v:1/(1+v*v)}[name]; T=[U,U*U,U*np.roll(U,1,axis=1),U+np.roll(U,1,axis=1)][mode]; print(fit(np.c_[np.ones(len(U)),U,f(w*T)],V).round(4))
