import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import product,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
def bias(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
p=s['panel_a'];U=bias(p['u']);i=np.linalg.norm(p['v']-U@fit(U,p['v']),axis=1)<3; A=fit(U[i],p['v'][i]); B=fit(bias(s['panel_b']['u']),s['panel_b']['v'])
a=lambda x:(x-A[0])/np.diag(A[1:]); aa=lambda x:bias(x)@A; b=lambda x:bias(x)@B; bb=lambda x:(x-B[0])@np.linalg.inv(B[1:]);
functions={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'rational':lambda z:z/(1+z*z),'softsign':lambda z:z/(1+abs(z)),'gauss':lambda z:np.exp(-z*z),'root':lambda z:z/np.sqrt(1+z*z)}
p=s['panel_c'];u=p['u']; models={}
for k in ('v1','v2'):
 v=p[k]; found=[]
 for name,fn in functions.items():
  for extra in (False,True):
   def feat(z,q):
    t=fn(z*q[:4]+q[4:]);return np.c_[bias(z),t] if extra else bias(t)
   for cut in (20,24):
    best=None
    for scale,phase in [(0.6,0),(1.4,0),(2.5,0),(1,0.5),(1,1)]:
     def obj(q):
      F=feat(u[:cut],q);return (F@fit(F,v[:cut])-v[:cut]).ravel()
     r=least_squares(obj,np.r_[np.full(4,scale),np.full(4,phase)],bounds=(np.r_[np.full(4,.06),np.full(4,-3.)],np.r_[np.full(4,6.),np.full(4,3.)]),max_nfev=100)
     e=np.mean(abs(r.fun));
     if best is None or e<best[0]:best=e,r.x
    e,q=best; w=fit(feat(u[:cut],q),v[:cut]); pred=lambda z,feat=feat,q=q,w=w:feat(z,q)@w
    if cut==20: report=(e,np.mean(abs(pred(u[20:])-v[20:])))
   found.append((report[1],name,extra,pred,q,w,report[0],e))
 found.sort(key=lambda r:r[0]);
 for ee,name,extra,pred,q,w,et,ef in found[:7]: print(k,name,extra,'panel tail,fit20,fit24',ee,et,ef,'q',q)
 models[k]=found[0][3]; print('best coefficients',found[0][5])
# Check terminal and overall validation for plausible compositions and two-head linear blocks.
x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; allf=[]
for pn,pf in [('x',lambda z:z),('a',a),('A',aa),('b',b),('ba',lambda z:b(a(z))),('bA',lambda z:b(aa(z))),('B',bb),('Ba',lambda z:bb(a(z)))]:
 z=pf(x)
 for cn,cf in [('none',lambda z:z),*models.items()]:
  zz=cf(z)
  for nn,w in [('id',zz),('b',b(zz)),('B',bb(zz))]:
   for tag,F in [('lin',bias(w)),('quad',poly(w,2)),('cub',poly(w,3)),('sep',np.c_[bias(w),w**2,w**3,w**4,w**5]),('joint',np.c_[bias(z),models['v1'](z),models['v2'](z)]),('joint2',np.c_[poly(z,2),poly(models['v1'](z),2)[:,5:],poly(models['v2'](z),2)[:,5:]])]:
    er=np.mean(abs(F[52:]@fit(F[:52],Y[:52])-Y[52:]),axis=0); allf.append((er[0],er[1:].mean(),er[-1],pn,cn,nn,tag))
for ix in (0,1):
 print('best',ix);print(*sorted(set(allf),key=lambda z:z[ix])[:18],sep='\n')
# Time-basis tests: a common decay plus sustained forcing/relaxation.
t=s['eval_times']; T=Y[:,1:]; _,_,V=np.linalg.svd(T,full_matrices=False); Q=V[:2].T
for tag,make,p0 in [('exp+texp',lambda p:np.c_[1-np.exp(-p[0]*t),t*np.exp(-p[0]*t)],[1.]),('exp+exp',lambda p:np.c_[1-np.exp(-p[0]*t),1-np.exp(-p[1]*t)],[.4,2.]),('exp+t',lambda p:np.c_[1-np.exp(-p[0]*t),t],[1.]),('exp+osc',lambda p:np.c_[1-np.exp(-p[0]*t),np.sin(p[1]*t)*np.exp(-p[2]*t)],[1.,1.,.5]),('exp+forced',lambda p:np.c_[1-np.exp(-p[0]*t),t-(1-np.exp(-p[1]*t))/p[1]],[1.,2.])]:
 def obj(p):
  F=make(p);return (F@fit(F,Q)-Q).ravel()
 o=least_squares(obj,p0,bounds=(.001,15)); F=make(o.x);print('time',tag,o.x,np.linalg.norm(o.fun),np.mean(abs(T-(F@fit(F,T.T)).T)))
